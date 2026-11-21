"""Hand-written optimizers must match torch.optim step by step."""
import copy

import pytest
import torch
from torch import nn

from training.optimizers import SGD, Adam, build_optimizer


def _pair(seed=0):
    """Two identical small models, so each optimizer drives its own copy."""
    torch.manual_seed(seed)
    a = nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 3))
    b = nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 3))
    b.load_state_dict(a.state_dict())
    return a, b


def _run(model_a, opt_a, model_b, opt_b, steps=25):
    torch.manual_seed(123)
    for _ in range(steps):
        x, y = torch.randn(16, 6), torch.randn(16, 3)
        for model, opt in ((model_a, opt_a), (model_b, opt_b)):
            opt.zero_grad()
            ((model(x) - y) ** 2).mean().backward()
            opt.step()
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(lr=0.05),
        dict(lr=0.05, momentum=0.9),
        dict(lr=0.05, momentum=0.9, weight_decay=0.01),
        dict(lr=0.05, momentum=0.9, nesterov=True),
    ],
)
def test_sgd_matches_torch(kwargs):
    a, b = _pair()
    _run(a, SGD(a.parameters(), **kwargs), b, torch.optim.SGD(b.parameters(), **kwargs))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(lr=1e-2),
        dict(lr=1e-2, betas=(0.8, 0.9)),
        dict(lr=1e-2, weight_decay=0.05),
    ],
)
def test_adam_matches_torch_adam(kwargs):
    a, b = _pair()
    _run(a, Adam(a.parameters(), **kwargs), b, torch.optim.Adam(b.parameters(), **kwargs))


def test_adamw_matches_torch_adamw():
    a, b = _pair()
    kwargs = dict(lr=1e-2, weight_decay=0.1)
    _run(a, Adam(a.parameters(), decoupled=True, **kwargs), b, torch.optim.AdamW(b.parameters(), **kwargs))


def test_adam_first_step_size_is_lr():
    """With bias correction, the first Adam update has magnitude ~lr whatever the gradient scale."""
    p = nn.Parameter(torch.zeros(3))
    opt = Adam([p], lr=0.1)
    p.grad = torch.tensor([1e-3, 1.0, 1e3])
    opt.step()
    assert torch.allclose(p.detach(), torch.full((3,), -0.1), atol=1e-4)


def test_skips_params_without_grad():
    p, q = nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(2))
    p.grad = torch.ones(2)
    for opt in (SGD([p, q], lr=0.1), Adam([p, q], lr=0.1)):
        opt.step()
    assert torch.equal(q.detach(), torch.ones(2))


def test_closure_is_evaluated():
    p = nn.Parameter(torch.tensor([2.0]))
    opt = Adam([p], lr=0.1)

    def closure():
        opt.zero_grad()
        loss = (p**2).sum()
        loss.backward()
        return loss

    assert opt.step(closure).item() == pytest.approx(4.0)


def test_state_dict_roundtrip_continues_identically():
    a, b = _pair()
    opt_a = Adam(a.parameters(), lr=1e-2)
    x, y = torch.randn(8, 6), torch.randn(8, 3)
    for _ in range(3):
        opt_a.zero_grad()
        ((a(x) - y) ** 2).mean().backward()
        opt_a.step()
    b.load_state_dict(a.state_dict())
    opt_b = Adam(b.parameters(), lr=1e-2)
    # deepcopy: load_state_dict may share tensors, which a real save/load round trip never does
    opt_b.load_state_dict(copy.deepcopy(opt_a.state_dict()))
    _run(a, opt_a, b, opt_b, steps=5)


def test_invalid_hyperparameters():
    p = [nn.Parameter(torch.zeros(1))]
    for bad in (dict(lr=-1), dict(betas=(1.0, 0.9)), dict(eps=0), dict(weight_decay=-1)):
        with pytest.raises(ValueError):
            Adam(p, **bad)
    for bad in (dict(lr=-1), dict(momentum=-1), dict(weight_decay=-1), dict(nesterov=True)):
        with pytest.raises(ValueError):
            SGD(p, **bad)


def test_build_optimizer_factory():
    params = [nn.Parameter(torch.zeros(1))]
    assert isinstance(build_optimizer("adam", params, 1e-3), Adam)
    assert build_optimizer("adamw", params, 1e-3).defaults["decoupled"] is True
    assert isinstance(build_optimizer("sgd", params, 1e-3), SGD)
    assert isinstance(build_optimizer("torch_adamw", params, 1e-3), torch.optim.AdamW)
    assert isinstance(build_optimizer("torch_sgd", params, 1e-3), torch.optim.SGD)
    with pytest.raises(ValueError):
        build_optimizer("rmsprop", params, 1e-3)


def test_optimizers_minimise_a_quadratic():
    for make in (lambda p: Adam(p, lr=0.1), lambda p: SGD(p, lr=0.1, momentum=0.9)):
        w = nn.Parameter(torch.tensor([5.0, -3.0]))
        opt = make([w])
        for _ in range(300):
            opt.zero_grad()
            (w**2).sum().backward()
            opt.step()
        assert w.abs().max() < 0.05
