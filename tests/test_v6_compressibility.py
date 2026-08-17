from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from minimind_v_bound.compression.description_length import (
    description_length_upper_bound,
)
from minimind_v_bound.compression.qat_quantization_v6 import (
    finalize_qat_quantization,
    initial_qat_centers,
    nearest_center_assignments,
    qat_ste_coordinate,
)
from minimind_v_bound.configuration_v6_compressibility import (
    V6_CANDIDATE_CODES,
    V6_CONFIGURATION_IDS,
    load_v6_registry,
    resolve_v6_candidate,
)
from minimind_v_bound.models.qat_projector_v6 import (
    QATIntrinsicSubLoRAProjectorV6,
    prepare_qat_projector_v6,
)
from minimind_v_bound.models.structured_projection_v6 import RoundedDoubleKronQRV6


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs/experiment_registry_v6_compressibility.yaml"


class TinyVLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(3, 3)
        self.vision_proj = nn.Sequential()
        self.vision_proj.add_module(
            "mlp",
            nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768, 768),
                nn.GELU(),
                nn.Linear(768, 768),
            ),
        )


def test_v6_registry_exactly_decodes_the_fifteen_candidate_table() -> None:
    registry = load_v6_registry(REGISTRY_PATH)
    candidates = [
        resolve_v6_candidate(registry, identifier, root=ROOT)
        for identifier in V6_CONFIGURATION_IDS
    ]
    assert tuple(candidate.candidate_id for candidate in candidates) == V6_CANDIDATE_CODES
    assert len(candidates) == 9
    assert len({candidate.output_directory for candidate in candidates}) == 9
    assert registry["description_length"]["structure_choice_bits"] == 4
    assert registry["description_length"][
        "no_separate_method_rank_dimension_or_level_cost"
    ] is True
    assert registry["certificate"]["model_count"] == 9
    assert {candidate.quantization_levels for candidate in candidates} == {11}
    assert registry["certificate"]["reuse_any_existing_certificate_sample"] is False


def test_v6_registry_rejects_structure_double_charging(tmp_path: Path) -> None:
    import yaml

    registry = load_v6_registry(REGISTRY_PATH)
    changed = copy.deepcopy(registry)
    changed["description_length"][
        "no_separate_method_rank_dimension_or_level_cost"
    ] = False
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="double charged"):
        load_v6_registry(path)


@pytest.mark.parametrize(
    ("output_dimension", "intrinsic_dimension"),
    [(3072, 256), (3072, 512), (12288, 512), (1_182_720, 512)],
)
def test_v6_projection_matches_pinned_official_forward(
    output_dimension: int,
    intrinsic_dimension: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "upstream/SubLoRA-bounds-for-LLMs"))
    from sublora.nn.projectors import RoundedDoubleKronQR as Official

    coordinate = torch.linspace(-0.1, 0.1, intrinsic_dimension)
    official = Official(output_dimension, intrinsic_dimension, [], [], seed=137) @ coordinate
    adapted = RoundedDoubleKronQRV6(output_dimension, intrinsic_dimension, 137)(
        coordinate
    )
    assert torch.allclose(adapted, official, rtol=0.0, atol=8e-9)


def test_v6_non_square_projection_backward_is_deterministic() -> None:
    first = torch.linspace(-0.1, 0.1, 512, requires_grad=True)
    second = first.detach().clone().requires_grad_(True)
    first_output = RoundedDoubleKronQRV6(3072, 512, 137)(first).square().sum()
    second_output = RoundedDoubleKronQRV6(3072, 512, 137)(second).square().sum()
    first_output.backward()
    second_output.backward()
    assert torch.equal(first_output, second_output)
    assert torch.equal(first.grad, second.grad)


def test_qat_forward_and_both_gradient_paths() -> None:
    vector = torch.tensor([-0.8, -0.1, 0.6], dtype=torch.float32, requires_grad=True)
    centers = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32, requires_grad=True)
    quantized, assignments = qat_ste_coordinate(vector, centers)
    assert assignments.tolist() == [0, 1, 2]
    assert torch.equal(quantized, torch.tensor([-1.0, 0.0, 1.0]))
    quantized.sum().backward()
    assert torch.equal(vector.grad, torch.ones_like(vector))
    assert torch.equal(centers.grad, torch.ones_like(centers))


def test_qat_tie_break_and_final_fp16_reassignment() -> None:
    vector = torch.tensor([0.0, 0.49, -0.49], dtype=torch.float32)
    centers = torch.tensor([-0.5, 0.5], dtype=torch.float32)
    assert nearest_center_assignments(vector, centers).tolist() == [0, 1, 0]
    final = finalize_qat_quantization(vector, centers)
    final.validate(dimension=3, levels=2)
    assert final.assignments.tolist() == [0, 1, 0]
    assert final.centers.dtype == np.float16


@pytest.mark.parametrize(
    ("parameterization", "rank", "dimension", "levels"),
    [("subspace", None, 256, 11), ("sublora", 1, 512, 17), ("sublora", 4, 1024, 11)],
)
def test_v6_wrapper_continuous_then_qat_semantics(
    parameterization: str, rank: int | None, dimension: int, levels: int
) -> None:
    torch.manual_seed(20260813)
    model = TinyVLM()
    wrapper = prepare_qat_projector_v6(
        model,
        parameterization=parameterization,
        intrinsic_dimension=dimension,
        quantization_levels=levels,
        rank=rank,
        lora_scale=32.0,
        subspace_seed=137,
        factor_init_seed=20260815,
    )
    assert torch.equal(wrapper.coordinate_for_forward(), wrapper.subspace_params)
    assert wrapper.quantization_centers.requires_grad
    with torch.no_grad():
        wrapper.subspace_params.copy_(torch.linspace(-0.02, 0.02, dimension))
    wrapper.initialize_qat_centers()
    wrapper.set_qat_forward(True)
    quantized = wrapper.coordinate_for_forward()
    assignments = wrapper.current_assignments()
    assert torch.equal(quantized, wrapper.quantization_centers[assignments])
    quantized.square().sum().backward()
    assert wrapper.subspace_params.grad is not None
    assert wrapper.quantization_centers.grad is not None


def test_v6_rank_one_initialization_hash_is_fixed() -> None:
    torch.manual_seed(20260813)
    wrapper = QATIntrinsicSubLoRAProjectorV6(
        TinyVLM().vision_proj,
        intrinsic_dimension=256,
        rank=1,
        quantization_levels=11,
        lora_scale=32.0,
        subspace_seed=137,
        factor_init_seed=20260815,
    )
    assert wrapper.factor_initialization_sha256() == (
        "db523f6009b03ad67ec28ecabbb8d5917c2fbfed110ccf65897b456a8a3b6d4c"
    )


def test_v6_candidate_id_cost_is_counted_once_in_description_length() -> None:
    counts = np.asarray([20, 40, 60, 40, 20, 10, 8, 6, 4, 3, 45], dtype=np.uint32)
    assert int(counts.sum()) == 256
    with_id = description_length_upper_bound(
        counts, dimension=256, structure_choice_bits=4
    )
    without_id = description_length_upper_bound(
        counts, dimension=256, structure_choice_bits=0
    )
    assert with_id["C_VLM_raw_upper_bits"] == (
        without_id["C_VLM_raw_upper_bits"] + 4
    )
    assert with_id["structure_choice_bits"] == 4


def test_qat_initialization_is_deterministic() -> None:
    vector = torch.linspace(-0.03, 0.04, 512, dtype=torch.float32)
    assert torch.equal(
        initial_qat_centers(vector, levels=17),
        initial_qat_centers(vector, levels=17),
    )
