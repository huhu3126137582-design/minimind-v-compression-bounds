from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from collections.abc import Iterator

import torch
from torch import Tensor, nn
from torch.func import functional_call

from minimind_v_bound.compression.qat_quantization_v6 import (
    initial_qat_centers,
    qat_ste_coordinate,
)

from .structured_projection_v6 import RoundedDoubleKronQRV6


QAT_PROJECTOR_VERSION = "v6-projector-coordinate-qat-v1"


class _QATCoordinateMixin:
    subspace_params: nn.Parameter
    quantization_centers: nn.Parameter
    quantization_levels: int

    def _initialize_qat_state(self, *, dimension: int, levels: int) -> None:
        self.quantization_levels = int(levels)
        if self.quantization_levels < 2:
            raise ValueError("QAT requires at least two centers")
        self.quantization_centers = nn.Parameter(
            torch.zeros(self.quantization_levels, dtype=torch.float32)
        )
        self.register_buffer(
            "qat_initialized", torch.tensor(False, dtype=torch.bool), persistent=True
        )
        self.qat_enabled_for_forward = False

    @torch.no_grad()
    def initialize_qat_centers(self) -> None:
        if bool(self.qat_initialized.item()):
            raise RuntimeError("QAT centers are already initialized")
        centers = initial_qat_centers(
            self.subspace_params.detach(), levels=self.quantization_levels
        )
        self.quantization_centers.copy_(centers)
        self.qat_initialized.fill_(True)

    def set_qat_forward(self, enabled: bool) -> None:
        if enabled and not bool(self.qat_initialized.item()):
            raise RuntimeError("cannot enable QAT before center initialization")
        self.qat_enabled_for_forward = bool(enabled)

    def coordinate_for_forward(self) -> Tensor:
        if not self.qat_enabled_for_forward:
            # Keep centers in the DDP graph before the QAT boundary with an
            # exact zero contribution and zero gradient.
            return self.subspace_params + 0.0 * self.quantization_centers.sum()
        coordinate, _ = qat_ste_coordinate(
            self.subspace_params.float(), self.quantization_centers.float()
        )
        return coordinate

    def current_assignments(self) -> Tensor:
        if not bool(self.qat_initialized.item()):
            raise RuntimeError("QAT centers have not been initialized")
        _, assignments = qat_ste_coordinate(
            self.subspace_params.detach().float(),
            self.quantization_centers.detach().float(),
        )
        return assignments


class QATIntrinsicProjectorV6(_QATCoordinateMixin, nn.Module):
    def __init__(
        self,
        base_projector: nn.Module,
        *,
        intrinsic_dimension: int,
        quantization_levels: int,
        subspace_seed: int,
    ) -> None:
        super().__init__()
        self.base_projector = base_projector
        for parameter in self.base_projector.parameters():
            parameter.requires_grad_(False)
        named = list(self.base_projector.named_parameters())
        if not named:
            raise ValueError("base Projector has no parameters")
        self.parameter_names = tuple(name for name, _ in named)
        self.parameter_shapes = tuple(tuple(parameter.shape) for _, parameter in named)
        self.parameter_numels = tuple(parameter.numel() for _, parameter in named)
        self.full_dimension = sum(self.parameter_numels)
        self.intrinsic_dimension = int(intrinsic_dimension)
        self.subspace_params = nn.Parameter(torch.zeros(self.intrinsic_dimension))
        self.projection = RoundedDoubleKronQRV6(
            self.full_dimension, self.intrinsic_dimension, subspace_seed
        )
        self.subspace_seed = int(subspace_seed)
        self._initialize_qat_state(
            dimension=self.intrinsic_dimension, levels=quantization_levels
        )

    def _unflatten_delta(self, flat_delta: Tensor) -> Iterator[Tensor]:
        offset = 0
        for shape, numel in zip(self.parameter_shapes, self.parameter_numels, strict=True):
            yield flat_delta[offset : offset + numel].reshape(shape)
            offset += numel
        if offset != flat_delta.numel():
            raise RuntimeError("projected parameter vector has the wrong length")

    def materialized_parameters(self) -> OrderedDict[str, Tensor]:
        device_type = self.subspace_params.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            flat_delta = self.projection(self.coordinate_for_forward().float())
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
            self.base_projector, self.materialized_parameters(), (inputs,), strict=True
        )


class QATIntrinsicSubLoRAProjectorV6(_QATCoordinateMixin, nn.Module):
    def __init__(
        self,
        base_projector: nn.Module,
        *,
        intrinsic_dimension: int,
        rank: int,
        quantization_levels: int,
        lora_scale: float,
        subspace_seed: int,
        factor_init_seed: int,
        target_linear_names: tuple[str, str] = ("mlp.1", "mlp.3"),
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if not math.isfinite(lora_scale) or lora_scale <= 0:
            raise ValueError("LoRA scale must be finite and positive")
        if tuple(target_linear_names) != ("mlp.1", "mlp.3"):
            raise ValueError("v6 only permits the frozen Linear targets")
        self.base_projector = base_projector
        for parameter in self.base_projector.parameters():
            parameter.requires_grad_(False)
        modules = dict(self.base_projector.named_modules())
        targets = []
        for name in target_linear_names:
            module = modules.get(name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"SubLoRA target is not Linear: {name}")
            targets.append((name, module))
        self.intrinsic_dimension = int(intrinsic_dimension)
        self.rank = int(rank)
        self.lora_scale = float(lora_scale)
        self.subspace_seed = int(subspace_seed)
        self.factor_init_seed = int(factor_init_seed)
        self.target_linear_names = tuple(target_linear_names)
        self.factor_names = tuple(
            item
            for target_name, _ in targets
            for item in (f"{target_name}.lora_A", f"{target_name}.lora_B")
        )
        self.factor_shapes = tuple(
            shape
            for _, module in targets
            for shape in (
                (self.rank, module.in_features),
                (module.out_features, self.rank),
            )
        )
        self.factor_numels = tuple(math.prod(shape) for shape in self.factor_shapes)
        self.lora_environment_dimension = sum(self.factor_numels)
        if self.intrinsic_dimension > self.lora_environment_dimension:
            raise ValueError("intrinsic dimension exceeds LoRA environment")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.factor_init_seed)
        initial_factors = []
        for _, module in targets:
            bound = 1.0 / math.sqrt(module.in_features)
            lora_a = torch.empty((self.rank, module.in_features)).uniform_(
                -bound, bound, generator=generator
            )
            lora_b = torch.zeros((module.out_features, self.rank))
            initial_factors.extend((lora_a, lora_b))
        self.register_buffer(
            "phi_0",
            torch.cat([factor.reshape(-1) for factor in initial_factors]).contiguous(),
            persistent=False,
        )
        self.subspace_params = nn.Parameter(torch.zeros(self.intrinsic_dimension))
        self.projection = RoundedDoubleKronQRV6(
            self.lora_environment_dimension,
            self.intrinsic_dimension,
            self.subspace_seed,
        )
        self._initialize_qat_state(
            dimension=self.intrinsic_dimension, levels=quantization_levels
        )

    @property
    def scaling(self) -> float:
        return self.lora_scale / self.rank

    def _unflatten_factors(self, flat: Tensor) -> Iterator[Tensor]:
        offset = 0
        for shape, numel in zip(self.factor_shapes, self.factor_numels, strict=True):
            yield flat[offset : offset + numel].reshape(shape)
            offset += numel
        if offset != flat.numel():
            raise RuntimeError("SubLoRA factor vector has the wrong length")

    def materialized_factors(self) -> OrderedDict[str, Tensor]:
        device_type = self.subspace_params.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            flat = self.phi_0 + self.projection(self.coordinate_for_forward().float())
        return OrderedDict(
            (name, factor)
            for name, factor in zip(
                self.factor_names, self._unflatten_factors(flat), strict=True
            )
        )

    def materialized_projector_parameters(self) -> OrderedDict[str, Tensor]:
        factors = self.materialized_factors()
        parameters = OrderedDict(self.base_projector.named_parameters())
        for target_name in self.target_linear_names:
            name = f"{target_name}.weight"
            base = parameters[name]
            update = (
                factors[f"{target_name}.lora_B"]
                @ factors[f"{target_name}.lora_A"]
            ) * self.scaling
            parameters[name] = base + update.to(device=base.device, dtype=base.dtype)
        return parameters

    def forward(self, inputs: Tensor) -> Tensor:
        return functional_call(
            self.base_projector,
            self.materialized_projector_parameters(),
            (inputs,),
            strict=True,
        )

    def factor_initialization_sha256(self) -> str:
        return hashlib.sha256(
            self.phi_0.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()


def prepare_qat_projector_v6(
    model: nn.Module,
    *,
    parameterization: str,
    intrinsic_dimension: int,
    quantization_levels: int,
    rank: int | None,
    lora_scale: float,
    subspace_seed: int,
    factor_init_seed: int,
) -> QATIntrinsicProjectorV6 | QATIntrinsicSubLoRAProjectorV6:
    if not hasattr(model, "vision_proj"):
        raise TypeError("model must expose vision_proj")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if parameterization == "subspace":
        if rank is not None:
            raise ValueError("Subspace candidate cannot specify LoRA rank")
        wrapper = QATIntrinsicProjectorV6(
            model.vision_proj,
            intrinsic_dimension=intrinsic_dimension,
            quantization_levels=quantization_levels,
            subspace_seed=subspace_seed,
        )
    elif parameterization == "sublora":
        if rank is None:
            raise ValueError("SubLoRA candidate requires a rank")
        wrapper = QATIntrinsicSubLoRAProjectorV6(
            model.vision_proj,
            intrinsic_dimension=intrinsic_dimension,
            rank=rank,
            quantization_levels=quantization_levels,
            lora_scale=lora_scale,
            subspace_seed=subspace_seed,
            factor_init_seed=factor_init_seed,
        )
    else:
        raise ValueError(f"unsupported v6 parameterization: {parameterization}")
    model.vision_proj = wrapper
    trainable = {
        name: parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    expected = {
        "vision_proj.subspace_params": intrinsic_dimension,
        "vision_proj.quantization_centers": quantization_levels,
    }
    if trainable != expected:
        raise RuntimeError(f"unexpected v6 trainable parameters: {trainable}")
    return wrapper
