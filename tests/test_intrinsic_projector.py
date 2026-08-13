from __future__ import annotations

import torch
from torch import nn

from minimind_v_bound.models.intrinsic_projector import IntrinsicProjector


def small_projector() -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(4),
        nn.Linear(4, 4),
        nn.GELU(),
        nn.Linear(4, 4),
    )


def test_zero_coordinate_exactly_matches_frozen_base() -> None:
    torch.manual_seed(7)
    base = small_projector()
    inputs = torch.randn(3, 4)
    expected = base(inputs).detach()
    wrapper = IntrinsicProjector(base, intrinsic_dimension=4, seed=137)
    actual = wrapper(inputs)
    assert torch.equal(actual, expected)


def test_only_subspace_coordinate_is_trainable_and_receives_gradient() -> None:
    torch.manual_seed(8)
    wrapper = IntrinsicProjector(small_projector(), intrinsic_dimension=4, seed=137)
    trainable = {
        name: parameter.numel()
        for name, parameter in wrapper.named_parameters()
        if parameter.requires_grad
    }
    assert trainable == {"subspace_params": 4}
    loss = wrapper(torch.randn(2, 4)).square().mean()
    loss.backward()
    assert wrapper.subspace_params.grad is not None
    assert torch.count_nonzero(wrapper.subspace_params.grad) > 0
    assert all(
        parameter.grad is None for parameter in wrapper.base_projector.parameters()
    )


def test_projection_does_not_mutate_base_parameter_objects() -> None:
    wrapper = IntrinsicProjector(small_projector(), intrinsic_dimension=4, seed=137)
    identities = [id(parameter) for parameter in wrapper.base_projector.parameters()]
    wrapper.subspace_params.data.fill_(0.1)
    wrapper(torch.randn(2, 4)).sum().backward()
    assert identities == [id(parameter) for parameter in wrapper.base_projector.parameters()]

