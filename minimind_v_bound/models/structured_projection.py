from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class RoundedDoubleKronQR(nn.Module):
    """Deterministic order-2 RDKronQR projection used by SubLoRA.

    This reproduces the random-number consumption and Kronecker mapping of
    ``RoundedDoubleKronQR`` at the pinned SubLoRA commit, while expressing the
    fixed factors as non-persistent PyTorch buffers.  The dense D-by-d matrix is
    never materialized.
    """

    algorithm_version = "pinned-sublora-c606ea6-order2-functional-v1"

    def __init__(self, output_dimension: int, intrinsic_dimension: int, seed: int):
        super().__init__()
        if output_dimension <= 0 or intrinsic_dimension <= 0:
            raise ValueError("projection dimensions must be positive")
        if intrinsic_dimension > output_dimension:
            raise ValueError("intrinsic dimension cannot exceed output dimension")

        self.output_dimension = int(output_dimension)
        self.intrinsic_dimension = int(intrinsic_dimension)
        self.seed = int(seed)
        self.rounded_output = math.isqrt(self.output_dimension)
        self.rounded_input = math.isqrt(self.intrinsic_dimension)
        main_output = self.rounded_output**2
        main_input = self.rounded_input**2
        self.main_output_dimension = main_output
        self.main_input_dimension = main_input

        # The official implementation isolates global RNG state.  fork_rng gives
        # the same property and, on CPU, the same seeded torch.randn/randint order.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            torch.randint(high=2**31, size=(1,))  # first derived seed (consumed)
            factors = []
            for _ in range(2):
                matrix = torch.randn(self.rounded_output, self.rounded_input)
                factor, _ = torch.linalg.qr(matrix, mode="reduced")
                factors.append(factor.contiguous())
                torch.randint(high=2**31, size=(1,))  # next derived seed

            tail_output = self.output_dimension - main_output
            tail_input = self.intrinsic_dimension - main_input
            # For S-4K, tail_input is zero.  Keeping the exact dense-tail branch
            # preserves the pinned implementation for any exact rounded side.
            if main_output == self.output_dimension or main_input == self.intrinsic_dimension:
                tail = torch.randn(tail_output, tail_input) / math.sqrt(
                    self.output_dimension
                )
            else:
                # The preregistered S-4K configuration never enters this branch.
                # Refuse it instead of silently substituting a different lazy
                # Gaussian algorithm whose backward must regenerate chunks.
                raise NotImplementedError(
                    "v1 supports configurations where D or d is a perfect square"
                )
            permutation = torch.randperm(self.output_dimension)

        self.register_buffer("factor_0", factors[0], persistent=False)
        self.register_buffer("factor_1", factors[1], persistent=False)
        self.register_buffer("tail", tail.contiguous(), persistent=False)
        self.register_buffer("permutation", permutation, persistent=False)

    @property
    def shape(self) -> tuple[int, int]:
        return self.output_dimension, self.intrinsic_dimension

    def _kron_matvec(self, vector: Tensor) -> Tensor:
        value = vector.reshape(self.rounded_input, self.rounded_input, 1)
        for axis, factor in enumerate((self.factor_0, self.factor_1)):
            front = torch.movedim(value, axis, 0)
            projected = factor @ front.reshape(self.rounded_input, -1)
            value = torch.movedim(
                projected.reshape(self.rounded_output, *front.shape[1:]), 0, axis
            )
        return value.reshape(self.main_output_dimension)

    def forward(self, vector: Tensor) -> Tensor:
        if vector.ndim != 1 or vector.shape[0] != self.intrinsic_dimension:
            raise ValueError(
                f"expected vector of shape ({self.intrinsic_dimension},), "
                f"got {tuple(vector.shape)}"
            )
        main = self._kron_matvec(vector[: self.main_input_dimension])
        tail_input = vector[self.main_input_dimension :]
        if self.tail.shape[0] == 0:
            tail_output = vector.new_empty((0,))
        elif self.tail.shape[1] == 0:
            tail_output = vector.new_zeros((self.tail.shape[0],))
        else:
            tail_output = self.tail @ tail_input
        unpermuted = torch.cat((main, tail_output), dim=0)
        return unpermuted[self.permutation]

