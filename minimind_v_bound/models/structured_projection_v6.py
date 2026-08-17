from __future__ import annotations

import math

import torch
from torch import Tensor, nn


PROJECTION_V6_VERSION = "pinned-sublora-c606ea6-order2-arbitrary-tail-v2"


class _SeededRandomMatvec(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, vector: Tensor, output_dimension: int, input_dimension: int, seed: int
    ) -> Tensor:
        ctx.info = (int(output_dimension), int(input_dimension), int(seed))
        devices = [vector.device.index] if vector.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if vector.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            matrix = torch.randn(
                output_dimension,
                input_dimension,
                device=vector.device,
                dtype=vector.dtype,
            )
        return matrix @ vector / math.sqrt(output_dimension)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        output_dimension, input_dimension, seed = ctx.info
        devices = [gradient.device.index] if gradient.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if gradient.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            matrix = torch.randn(
                output_dimension,
                input_dimension,
                device=gradient.device,
                dtype=gradient.dtype,
            )
        return matrix.T @ gradient / math.sqrt(output_dimension), None, None, None


class RoundedDoubleKronQRV6(nn.Module):
    """Pinned official order-2 RDKronQR, including its non-square lazy tail."""

    algorithm_version = PROJECTION_V6_VERSION

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
        self.main_output_dimension = self.rounded_output**2
        self.main_input_dimension = self.rounded_input**2
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            torch.randint(high=2**31, size=(1,))
            factors = []
            for _ in range(2):
                matrix = torch.randn(self.rounded_output, self.rounded_input)
                factor, _ = torch.linalg.qr(matrix, mode="reduced")
                factors.append(factor.contiguous())
                next_seed = int(torch.randint(high=2**31, size=(1,))[0])
            self.tail_seed = next_seed
            tail_output = self.output_dimension - self.main_output_dimension
            tail_input = self.intrinsic_dimension - self.main_input_dimension
            if self.main_output_dimension == self.output_dimension or (
                self.main_input_dimension == self.intrinsic_dimension
            ):
                fixed_tail = torch.randn(tail_output, tail_input) / math.sqrt(
                    self.output_dimension
                )
                self.tail_is_lazy_random = False
            else:
                fixed_tail = torch.empty(0, 0)
                self.tail_is_lazy_random = True
            permutation = torch.randperm(self.output_dimension)
        self.tail_output_dimension = tail_output
        self.tail_input_dimension = tail_input
        self.register_buffer("factor_0", factors[0], persistent=False)
        self.register_buffer("factor_1", factors[1], persistent=False)
        self.register_buffer("fixed_tail", fixed_tail.contiguous(), persistent=False)
        self.register_buffer("permutation", permutation, persistent=False)

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
            raise ValueError(f"expected vector of shape ({self.intrinsic_dimension},)")
        main = self._kron_matvec(vector[: self.main_input_dimension])
        tail_input = vector[self.main_input_dimension :]
        if self.tail_output_dimension == 0:
            tail = vector.new_empty((0,))
        elif self.tail_input_dimension == 0:
            tail = vector.new_zeros((self.tail_output_dimension,))
        elif self.tail_is_lazy_random:
            tail = _SeededRandomMatvec.apply(
                tail_input,
                self.tail_output_dimension,
                self.tail_input_dimension,
                self.tail_seed,
            )
        else:
            tail = self.fixed_tail @ tail_input
        return torch.cat((main, tail), dim=0)[self.permutation]
