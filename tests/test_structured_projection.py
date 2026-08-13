from __future__ import annotations

import torch

from minimind_v_bound.models.structured_projection import RoundedDoubleKronQR


def test_projection_is_deterministic_and_rng_isolated() -> None:
    torch.manual_seed(111)
    expected_next = torch.randn(5)
    torch.manual_seed(111)
    first = RoundedDoubleKronQR(100, 16, seed=137)
    actual_next = torch.randn(5)
    second = RoundedDoubleKronQR(100, 16, seed=137)
    vector = torch.linspace(-1.0, 1.0, 16)
    assert torch.equal(actual_next, expected_next)
    assert torch.equal(first(vector), second(vector))


def test_s4k_projection_has_expected_shape_norm_and_gradient() -> None:
    projection = RoundedDoubleKronQR(1_182_720, 4_096, seed=137)
    vector = torch.randn(4_096, requires_grad=True)
    output = projection(vector)
    assert output.shape == (1_182_720,)
    # d=4096 is a perfect square, so the Kronecker block has orthonormal
    # columns; rounding D only introduces output coordinates fixed to zero.
    assert torch.allclose(output.norm(), vector.norm(), rtol=2e-5, atol=2e-5)
    output.square().sum().backward()
    assert vector.grad is not None
    assert torch.allclose(vector.grad, 2 * vector.detach(), rtol=3e-5, atol=3e-5)

