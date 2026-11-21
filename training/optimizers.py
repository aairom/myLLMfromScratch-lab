"""Hand-written optimizers: SGD with momentum and Adam / AdamW.

Training minimises a loss ``L(theta)``. Backpropagation gives the gradient
``g = dL/dtheta``; an *optimizer* decides how to turn ``g`` into a parameter
update. Both classes follow the ``torch.optim.Optimizer`` interface (so they
support ``zero_grad``, ``state_dict`` ...) but implement ``step`` explicitly.
The unit tests check them against ``torch.optim`` step by step.

SGD with momentum
    buf   = momentum * buf + g                 (a running "velocity")
    theta = theta - lr * buf

Adam (Kingma & Ba, 2015)
    m = beta1 * m + (1 - beta1) * g            (1st moment: smoothed gradient)
    v = beta2 * v + (1 - beta2) * g^2          (2nd moment: smoothed squared gradient)
    m_hat = m / (1 - beta1^t)                  (bias correction: m, v start at 0)
    v_hat = v / (1 - beta2^t)
    theta = theta - lr * m_hat / (sqrt(v_hat) + eps)

    Dividing by sqrt(v_hat) gives every parameter its own effective step size:
    parameters with consistently large gradients move less, rare ones move more.

Weight decay
    * L2 (``decoupled=False``): add ``wd * theta`` to the gradient (Adam).
    * Decoupled (``decoupled=True``, i.e. AdamW): shrink ``theta`` directly by
      ``lr * wd`` before the update, independent of the gradient scale.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.optim import Optimizer

OPTIMIZER_NAMES = ("adam", "adamw", "sgd", "torch_adamw", "torch_sgd")


class SGD(Optimizer):
    """Stochastic gradient descent with optional (Nesterov) momentum and L2 decay."""

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        if lr < 0:
            raise ValueError("lr must be >= 0")
        if momentum < 0:
            raise ValueError("momentum must be >= 0")
        if weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires momentum > 0")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g + wd * p  # L2 regularisation folded into the gradient
                if mom != 0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = g.clone()  # first step: buf = g
                    else:
                        state["momentum_buffer"].mul_(mom).add_(g)
                    buf = state["momentum_buffer"]
                    g = g + mom * buf if group["nesterov"] else buf
                p.add_(g, alpha=-lr)
        return loss


class Adam(Optimizer):
    """Adam; set ``decoupled=True`` for AdamW-style weight decay."""

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decoupled: bool = False,
    ) -> None:
        if lr < 0:
            raise ValueError("lr must be >= 0")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError("betas must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be > 0")
        if weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, decoupled=decoupled)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:  # lazy state initialisation
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)  # m
                    state["exp_avg_sq"] = torch.zeros_like(p)  # v
                state["step"] += 1
                t = state["step"]

                if wd != 0:
                    if group["decoupled"]:
                        p.mul_(1 - lr * wd)  # AdamW: shrink weights directly
                    else:
                        g = g + wd * p  # Adam: L2 penalty enters the gradient

                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                bias1 = 1 - beta1**t
                bias2 = 1 - beta2**t
                denom = (v.sqrt() / math.sqrt(bias2)).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bias1)
        return loss


def build_optimizer(
    name: str,
    params: Iterable,
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    betas: tuple[float, float] = (0.9, 0.999),
) -> Optimizer:
    """Create an optimizer by name.

    ``adam`` / ``adamw`` / ``sgd`` are the hand-written classes above;
    ``torch_adamw`` / ``torch_sgd`` are the reference implementations from PyTorch.
    """
    params = list(params)
    if name == "adam":
        return Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    if name == "adamw":
        return Adam(params, lr=lr, betas=betas, weight_decay=weight_decay, decoupled=True)
    if name == "sgd":
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "torch_adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    if name == "torch_sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}; choose from {OPTIMIZER_NAMES}")
