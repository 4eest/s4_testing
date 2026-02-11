import copy

import pytest
import torch

import src.utils as utils
import src.utils.registry as registry


def _build_model(device):
    model_cfg = {
        '_name_': 'model',
        'd_model': 32,
        'n_layers': 2,
        'transposed': False,
        'prenorm': True,
        'dropout': 0.0,
        'tie_dropout': False,
        'track_norms': False,
        'residual': None,
        'norm': None,
        'layer': {
            '_name_': 'qkv_s4_multilinear',
            'd_branch': 32,
            'merge_type': 'gated',
            'share_ssm_weights': False,
            'merge_pre_norm': False,
            'prenorm': True,
            'dropout': 0.0,
            'tie_dropout': False,
            'ssm_kwargs': {
                'layer': 'fftconv',
                'mode': 'diag',
                'bidirectional': False,
                'channels': 1,
                'activation': 'gelu',
                'final_act': None,
                'gate': None,
                'bottleneck': None,
                'l_max': 16,
            },
        },
    }

    model = utils.instantiate(registry.model, copy.deepcopy(model_cfg))
    return model.to(device)


def _assert_model_structure(output, states, n_layers):
    assert isinstance(states, list)
    assert len(states) == n_layers
    assert isinstance(output, torch.Tensor)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_qkv_s4_multilinear_smoke(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available in this environment.')

    torch.manual_seed(0)
    model = _build_model(device)

    x = torch.randn(2, 16, 32, device=device)
    output, states = model(x)

    _assert_model_structure(output, states, n_layers=2)
    assert output.shape == x.shape

    # Verify return structure matches the standard SequenceModel with S4 layer.
    baseline_cfg = {
        '_name_': 'model',
        'd_model': 32,
        'n_layers': 2,
        'transposed': False,
        'prenorm': True,
        'dropout': 0.0,
        'tie_dropout': False,
        'track_norms': False,
        'residual': 'R',
        'norm': 'layer',
        'layer': {
            '_name_': 's4',
            'layer': 'fftconv',
            'mode': 'diag',
            'bidirectional': False,
            'channels': 1,
            'gate': None,
            'bottleneck': None,
            'final_act': None,
            'activation': 'gelu',
            'l_max': 16,
        },
    }
    baseline = utils.instantiate(registry.model, copy.deepcopy(baseline_cfg)).to(device)
    baseline_output, baseline_states = baseline(x)

    assert isinstance(baseline_output, torch.Tensor)
    assert isinstance(baseline_states, list)
    assert len(baseline_states) == len(states)
