from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from collections.abc import Iterator

import torch
from torch import Tensor, nn
from torch.func import functional_call

from .structured_projection import RoundedDoubleKronQR


SUBLORA_PARAMETERIZATION_VERSION = "projector-sublora-two-linear-functional-v1"


class IntrinsicSubLoRAProjector(nn.Module):
    """Apply a fixed random subspace to both LoRA factors of two Linear layers.

    The theorem-side coordinate is ``u`` and the factor vector is
    ``phi(u) = phi_0 + P u``.  The resulting Projector weights are nonlinear in
    ``u`` because each update is ``(lora_scale / rank) * B(u) @ A(u)``.
    """

    def __init__(
        self,
        base_projector: nn.Module,
        *,
        intrinsic_dimension: int,
        rank: int,
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
        self.base_projector = base_projector
        for parameter in self.base_projector.parameters():
            parameter.requires_grad_(False)

        modules = dict(self.base_projector.named_modules())
        if tuple(target_linear_names) != ("mlp.1", "mlp.3"):
            raise ValueError("v3 only permits the preregistered Linear targets")
        targets: list[tuple[str, nn.Linear]] = []
        for name in target_linear_names:
            module = modules.get(name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"SubLoRA target is not a Linear layer: {name}")
            targets.append((name, module))
        if any(module.bias is None for _, module in targets):
            raise ValueError("the pinned Projector Linear layers must have fixed biases")

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
            raise ValueError("intrinsic dimension exceeds the LoRA factor environment")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.factor_init_seed)
        initial_factors: list[Tensor] = []
        for _, module in targets:
            bound = 1.0 / math.sqrt(module.in_features)
            lora_a = torch.empty(
                (self.rank, module.in_features), dtype=torch.float32
            ).uniform_(-bound, bound, generator=generator)
            lora_b = torch.zeros(
                (module.out_features, self.rank), dtype=torch.float32
            )
            initial_factors.extend((lora_a, lora_b))
        phi_0 = torch.cat([factor.reshape(-1) for factor in initial_factors])
        self.register_buffer("phi_0", phi_0.contiguous(), persistent=False)

        self.subspace_params = nn.Parameter(torch.zeros(self.intrinsic_dimension))
        self.projection = RoundedDoubleKronQR(
            self.lora_environment_dimension,
            self.intrinsic_dimension,
            self.subspace_seed,
        )

    @property
    def scaling(self) -> float:
        return self.lora_scale / self.rank

    def _unflatten_factors(self, flat_factors: Tensor) -> Iterator[Tensor]:
        offset = 0
        for shape, numel in zip(self.factor_shapes, self.factor_numels, strict=True):
            yield flat_factors[offset : offset + numel].reshape(shape)
            offset += numel
        if offset != flat_factors.numel():
            raise RuntimeError("SubLoRA factor vector has the wrong length")

    def materialized_factors(self) -> OrderedDict[str, Tensor]:
        device_type = self.subspace_params.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            delta = self.projection(self.subspace_params.float())
            flat_factors = self.phi_0 + delta
        return OrderedDict(
            (name, factor)
            for name, factor in zip(
                self.factor_names,
                self._unflatten_factors(flat_factors),
                strict=True,
            )
        )

    def materialized_projector_parameters(self) -> OrderedDict[str, Tensor]:
        factors = self.materialized_factors()
        parameters = OrderedDict(self.base_projector.named_parameters())
        for target_name in self.target_linear_names:
            weight_name = f"{target_name}.weight"
            base_weight = parameters[weight_name]
            lora_a = factors[f"{target_name}.lora_A"]
            lora_b = factors[f"{target_name}.lora_B"]
            update = (lora_b @ lora_a) * self.scaling
            parameters[weight_name] = base_weight + update.to(
                device=base_weight.device, dtype=base_weight.dtype
            )
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

    def trainable_state_dict(self) -> dict[str, Tensor]:
        return {"subspace_params": self.subspace_params.detach().cpu().clone()}


def prepare_intrinsic_sublora_projector(
    model: nn.Module,
    *,
    intrinsic_dimension: int,
    rank: int,
    lora_scale: float,
    subspace_seed: int,
    factor_init_seed: int,
) -> IntrinsicSubLoRAProjector:
    """Freeze MiniMind-V and replace only ``vision_proj`` with SubLoRA."""
    if not hasattr(model, "vision_proj"):
        raise TypeError("model must expose a vision_proj module")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    wrapper = IntrinsicSubLoRAProjector(
        model.vision_proj,
        intrinsic_dimension=intrinsic_dimension,
        rank=rank,
        lora_scale=lora_scale,
        subspace_seed=subspace_seed,
        factor_init_seed=factor_init_seed,
    )
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
