from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator

import torch
from torch import Tensor, nn
from torch.func import functional_call

from .structured_projection import RoundedDoubleKronQR


class IntrinsicProjector(nn.Module):
    """Represent a frozen Projector as omega_0 + P u without mutation."""

    def __init__(
        self,
        base_projector: nn.Module,
        intrinsic_dimension: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.base_projector = base_projector
        for parameter in self.base_projector.parameters():
            parameter.requires_grad_(False)

        named_parameters = list(self.base_projector.named_parameters())
        if not named_parameters:
            raise ValueError("base Projector has no parameters")
        self.parameter_names = tuple(name for name, _ in named_parameters)
        self.parameter_shapes = tuple(tuple(parameter.shape) for _, parameter in named_parameters)
        self.parameter_numels = tuple(parameter.numel() for _, parameter in named_parameters)
        self.full_dimension = sum(self.parameter_numels)
        self.intrinsic_dimension = int(intrinsic_dimension)
        self.subspace_params = nn.Parameter(torch.zeros(self.intrinsic_dimension))
        self.projection = RoundedDoubleKronQR(
            self.full_dimension, self.intrinsic_dimension, seed
        )

    def _unflatten_delta(self, flat_delta: Tensor) -> Iterator[Tensor]:
        offset = 0
        for shape, numel in zip(self.parameter_shapes, self.parameter_numels, strict=True):
            yield flat_delta[offset : offset + numel].reshape(shape)
            offset += numel
        if offset != flat_delta.numel():
            raise RuntimeError("projected parameter vector has the wrong length")

    def materialized_parameters(self) -> OrderedDict[str, Tensor]:
        # Keep the structured projection itself in FP32 under mixed precision.
        device_type = self.subspace_params.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            flat_delta = self.projection(self.subspace_params.float())
        parameters = OrderedDict()
        base_parameters = dict(self.base_projector.named_parameters())
        for name, delta in zip(
            self.parameter_names, self._unflatten_delta(flat_delta), strict=True
        ):
            base = base_parameters[name]
            parameters[name] = base + delta.to(device=base.device, dtype=base.dtype)
        return parameters

    def forward(self, inputs: Tensor) -> Tensor:
        return functional_call(
            self.base_projector,
            self.materialized_parameters(),
            (inputs,),
            strict=True,
        )

    def trainable_state_dict(self) -> dict[str, Tensor]:
        return {"subspace_params": self.subspace_params.detach().cpu().clone()}


def prepare_intrinsic_projector(
    model: nn.Module, *, intrinsic_dimension: int, seed: int
) -> IntrinsicProjector:
    """Freeze a MiniMind-V model and replace only ``vision_proj``."""
    if not hasattr(model, "vision_proj"):
        raise TypeError("model must expose a vision_proj module")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    wrapper = IntrinsicProjector(model.vision_proj, intrinsic_dimension, seed)
    model.vision_proj = wrapper
    trainable = {
        name: parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    expected = {"vision_proj.subspace_params": intrinsic_dimension}
    if trainable != expected:
        raise RuntimeError(f"unexpected trainable parameters: {trainable}")
    return wrapper


def set_projector_training_mode(model: nn.Module) -> None:
    """Enable training forward semantics while keeping SigLIP in eval mode."""
    model.train()
    if not hasattr(model, "vision_encoder") or model.vision_encoder is None:
        raise RuntimeError("frozen vision encoder is unavailable")
    model.vision_encoder.eval()
