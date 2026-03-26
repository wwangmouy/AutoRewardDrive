from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch

try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call  # type: ignore


def named_parameter_dict(module: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, param) for name, param in module.named_parameters())


def named_buffer_dict(module: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, buf) for name, buf in module.named_buffers())


def functional_module_call(
    module: torch.nn.Module,
    params: Mapping[str, torch.Tensor],
    *args,
    **kwargs,
) -> torch.Tensor:
    state = OrderedDict(named_buffer_dict(module))
    state.update(params)
    return functional_call(module, state, args, kwargs)


def gradient_step(
    loss: torch.Tensor,
    params: Mapping[str, torch.Tensor],
    step_size: float,
    create_graph: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    gradients = torch.autograd.grad(
        loss,
        list(params.values()),
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=False,
    )
    updated = OrderedDict()
    for (name, value), grad in zip(params.items(), gradients):
        updated[name] = value - step_size * grad
    return updated


def optimize_actions_with_reward(
    reward_net: torch.nn.Module,
    params: Mapping[str, torch.Tensor],
    features: torch.Tensor,
    init_actions: torch.Tensor,
    steps: int,
    step_size: float,
) -> torch.Tensor:
    actions = init_actions.detach()
    for _ in range(max(int(steps), 1)):
        actions = actions.clone().detach().requires_grad_(True)
        reward = functional_module_call(reward_net, params, features, actions).mean()
        action_grad = torch.autograd.grad(reward, actions, create_graph=True, retain_graph=True)[0]
        actions = torch.clamp(actions + step_size * action_grad, -1.0, 1.0)
    return actions


def l2_from_params(params: Mapping[str, torch.Tensor]) -> torch.Tensor:
    total = None
    for value in params.values():
        current = value.pow(2).mean()
        total = current if total is None else total + current
    if total is None:
        raise ValueError("Expected at least one parameter when computing l2_from_params")
    return total
