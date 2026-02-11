"""QKV-parallel S4 branches with multilinear merge."""

from functools import partial

import torch
import torch.nn as nn
from einops import rearrange

from src.models.nn import DropoutNd
from src.models.sequence.base import SequenceModule
from src.models.sequence.modules.s4block import S4Block


class QKVSSMMultilinearBlock(SequenceModule):
    """Parallel Q/K/V SSM branches merged with multilinear interactions.

    This layer follows the integrated sequence layer API and returns ``(y, state)``.
    """

    def __init__(
        self,
        d_model,
        d_branch=None,
        merge_type='gated',
        share_ssm_weights=False,
        merge_pre_norm=False,
        prenorm=True,
        dropout=0.0,
        tie_dropout=False,
        transposed=False,
        ssm_kwargs=None,
    ):
        super().__init__()

        if merge_type not in {'gated', 'rank1'}:
            raise ValueError(f"Unsupported merge_type={merge_type}")

        self.d_model = d_model
        self.d_branch = d_model if d_branch is None else d_branch
        self.merge_type = merge_type
        self.share_ssm_weights = share_ssm_weights
        self.merge_pre_norm = merge_pre_norm
        self.prenorm = prenorm
        self.transposed = transposed

        self.norm = nn.LayerNorm(d_model) if merge_pre_norm else nn.Identity()

        self.q_proj = nn.Linear(d_model, self.d_branch)
        self.k_proj = nn.Linear(d_model, self.d_branch)
        self.v_proj = nn.Linear(d_model, self.d_branch)

        ssm_kwargs = {} if ssm_kwargs is None else dict(ssm_kwargs)
        ssm_kwargs.pop('_name_', None)
        ssm_kwargs.pop('d_model', None)
        ssm_kwargs.pop('transposed', None)

        self.q_ssm = S4Block(d_model=self.d_branch, transposed=False, **ssm_kwargs)
        if self.share_ssm_weights:
            self.k_ssm = self.q_ssm
            self.v_ssm = self.q_ssm
        else:
            self.k_ssm = S4Block(d_model=self.d_branch, transposed=False, **ssm_kwargs)
            self.v_ssm = S4Block(d_model=self.d_branch, transposed=False, **ssm_kwargs)

        self.uq = nn.Linear(self.d_branch, self.d_branch)
        self.uk = nn.Linear(self.d_branch, self.d_branch)
        self.uv = nn.Linear(self.d_branch, self.d_branch)

        self.out_proj = nn.Linear(self.d_branch, d_model)

        dropout_cls = partial(DropoutNd, transposed=False) if tie_dropout else nn.Dropout
        self.drop = dropout_cls(dropout) if dropout > 0.0 else nn.Identity()

    def _split_state(self, state):
        if state is None:
            return None, None, None
        if isinstance(state, (tuple, list)) and len(state) == 3:
            return state[0], state[1], state[2]
        raise ValueError('Expected state to be None or a 3-tuple/list of branch states.')

    def _merge(self, q2, k2, v2):
        uq = self.uq(q2)
        uk = self.uk(k2)
        uv = self.uv(v2)
        if self.merge_type == 'gated':
            return uv * torch.sigmoid(uq + uk)
        return uq * uk * uv

    def forward(self, x, state=None, **kwargs):
        if self.transposed:
            x = rearrange(x, 'b d ... -> b ... d')

        residual = x
        y = self.norm(x) if self.prenorm else x

        q = self.q_proj(y)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q_state, k_state, v_state = self._split_state(state)
        q2, next_q_state = self.q_ssm(q, state=q_state, **kwargs)
        k2, next_k_state = self.k_ssm(k, state=k_state, **kwargs)
        v2, next_v_state = self.v_ssm(v, state=v_state, **kwargs)

        merged = self._merge(q2, k2, v2)
        out = residual + self.drop(self.out_proj(merged))

        if not self.prenorm:
            out = self.norm(out)

        if self.transposed:
            out = rearrange(out, 'b ... d -> b d ...')

        return out, (next_q_state, next_k_state, next_v_state)

    def step(self, x, state, **kwargs):
        q_state, k_state, v_state = self._split_state(state)

        y = self.norm(x) if self.prenorm else x
        q = self.q_proj(y)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q2, next_q_state = self.q_ssm.step(q, q_state, **kwargs)
        k2, next_k_state = self.k_ssm.step(k, k_state, **kwargs)
        v2, next_v_state = self.v_ssm.step(v, v_state, **kwargs)

        merged = self._merge(q2, k2, v2)
        out = x + self.drop(self.out_proj(merged))

        if not self.prenorm:
            out = self.norm(out)

        return out, (next_q_state, next_k_state, next_v_state)

    def default_state(self, *batch_shape, device=None):
        q_state = self.q_ssm.default_state(*batch_shape, device=device)
        k_state = self.k_ssm.default_state(*batch_shape, device=device)
        v_state = self.v_ssm.default_state(*batch_shape, device=device)
        return (q_state, k_state, v_state)

    @property
    def d_output(self):
        return self.d_model

    @property
    def d_state(self):
        branch_d_state = self.q_ssm.d_state
        if branch_d_state is None:
            return None
        return 3 * branch_d_state

    @property
    def state_to_tensor(self):
        def fn(state):
            q_state, k_state, v_state = self._split_state(state)
            q_tensor = self.q_ssm.state_to_tensor(q_state)
            k_tensor = self.k_ssm.state_to_tensor(k_state)
            v_tensor = self.v_ssm.state_to_tensor(v_state)
            return torch.cat((q_tensor, k_tensor, v_tensor), dim=-1)

        return fn
