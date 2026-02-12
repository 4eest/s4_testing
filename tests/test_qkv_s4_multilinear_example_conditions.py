"""Example.py-style checks for QKV S4 multilinear architecture.

This mirrors the key training/runtime conditions used by `example.py`:
- input shape (B, L, d_input)
- encoder -> stacked residual sequence blocks -> mean pooling -> classifier
- optimizer setup that preserves S4 special hyperparameters via `_optim`
"""

import copy

import torch
import torch.nn as nn
import torch.optim as optim

import src.utils as utils
import src.utils.registry as registry


def setup_optimizer_like_example(model: nn.Module, lr: float, weight_decay: float, epochs: int):
    """Replicate optimizer logic from `example.py` for S4 special params."""
    all_parameters = list(model.parameters())

    params = [p for p in all_parameters if not hasattr(p, "_optim")]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]

    for hp in hps:
        hp_params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group({"params": hp_params, **hp})

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    return optimizer, scheduler


class ExampleStyleQKVModel(nn.Module):
    """Minimal classifier wrapper with example.py-style structure."""

    def __init__(
        self,
        d_input: int,
        d_output: int,
        d_model: int,
        n_layers: int,
        prenorm: bool,
        dropout: float,
    ):
        super().__init__()
        self.encoder = nn.Linear(d_input, d_model)
        model_cfg = {
            "_name_": "model",
            "d_model": d_model,
            "n_layers": n_layers,
            "transposed": False,
            "prenorm": prenorm,
            "dropout": dropout,
            "tie_dropout": False,
            "track_norms": False,
            # Keep residual/norm active to match integrated backbone conventions.
            "residual": "R",
            "norm": "layer",
            "layer": {
                "_name_": "qkv_s4_multilinear",
                "d_branch": d_model,
                "merge_type": "gated",
                "share_ssm_weights": False,
                "merge_pre_norm": False,
                "prenorm": prenorm,
                "dropout": dropout,
                "tie_dropout": False,
                "ssm_kwargs": {
                    "layer": "fftconv",
                    "mode": "diag",
                    "bidirectional": False,
                    "channels": 1,
                    "activation": "gelu",
                    "final_act": None,
                    "gate": None,
                    "bottleneck": None,
                    "l_max": 16,
                    "lr": {"dt": 0.001, "A": 0.001, "B": 0.001},
                    "wd": 0.0,
                },
            },
        }
        self.backbone = utils.instantiate(registry.model, copy.deepcopy(model_cfg))
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        x = self.encoder(x)
        x, _ = self.backbone(x)
        x = x.mean(dim=1)
        return self.decoder(x)


def test_qkv_example_conditions_cpu():
    torch.manual_seed(0)

    model = ExampleStyleQKVModel(
        d_input=3,
        d_output=10,
        d_model=32,
        n_layers=2,
        prenorm=True,
        dropout=0.1,
    )

    optimizer, scheduler = setup_optimizer_like_example(
        model, lr=0.01, weight_decay=0.01, epochs=5
    )

    x = torch.randn(2, 16, 3)
    y = torch.randint(0, 10, (2,))
    logits = model(x)

    assert logits.shape == (2, 10)

    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Validate S4 special optimizer metadata is preserved, like in example.py behavior.
    special_params = [p for p in model.parameters() if hasattr(p, "_optim")]
    assert len(special_params) > 0
    assert len(optimizer.param_groups) > 1

    # Check gradients flowed through model.
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    assert grad_count > 0
