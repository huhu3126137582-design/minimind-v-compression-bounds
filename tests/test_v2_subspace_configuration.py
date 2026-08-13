from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from minimind_v_bound.compression.description_length import (
    description_length_upper_bound,
)
from minimind_v_bound.configuration import (
    load_v2_subspace_registry,
    resolve_subspace_candidate,
)
from minimind_v_bound.models.structured_projection import RoundedDoubleKronQR
from quantize_final_subspace_v2 import validate_v2_final_checkpoint
from train_subspace_v2 import build_training_contract


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs/experiment_registry_v2_subspace.yaml"


def test_v2_registry_discloses_timing_and_fixes_three_candidates() -> None:
    registry = load_v2_subspace_registry(REGISTRY_PATH)
    candidates = [
        resolve_subspace_candidate(registry, identifier, root=ROOT)
        for identifier in ("S-1K", "S-4K", "S-16K")
    ]

    assert [candidate.intrinsic_dimension for candidate in candidates] == [
        1024,
        4096,
        16384,
    ]
    assert [candidate.action for candidate in candidates] == [
        "train_new",
        "import_frozen_v1",
        "train_new",
    ]
    assert registry["description_length"]["structure_choice_bits"] == 2
    assert registry["certificate"]["model_count"] == 3
    assert registry["certificate"]["reuse_v1_certificate_sample"] is False
    assert registry["scope_and_timing"][
        "v1_locked_test_was_observed_before_this_registry"
    ] is True


def test_v2_registry_rejects_undercharged_candidate_choice(tmp_path: Path) -> None:
    registry = load_v2_subspace_registry(REGISTRY_PATH)
    changed = copy.deepcopy(registry)
    changed["description_length"]["structure_choice_bits"] = 1
    path = tmp_path / "undercharged.yaml"
    import yaml

    path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="structure choice"):
        load_v2_subspace_registry(path)


@pytest.mark.parametrize("dimension", [1024, 16384])
def test_v2_projection_dimensions_are_deterministic_orthonormal_and_differentiable(
    dimension: int,
) -> None:
    first = RoundedDoubleKronQR(1_182_720, dimension, seed=137)
    second = RoundedDoubleKronQR(1_182_720, dimension, seed=137)
    coordinate = torch.linspace(-1.0, 1.0, dimension, requires_grad=True)
    projected = first(coordinate)

    assert projected.shape == (1_182_720,)
    assert torch.equal(projected.detach(), second(coordinate.detach()))
    # The pinned reference uses FP32 QR factors; the two Kronecker applications
    # accumulate roughly 4e-5 relative norm error at these wider factor sizes.
    assert torch.allclose(projected.norm(), coordinate.norm(), rtol=6e-5, atol=6e-5)
    projected.square().sum().backward()
    assert coordinate.grad is not None
    assert torch.allclose(coordinate.grad, 2 * coordinate.detach(), rtol=8e-5, atol=8e-5)


@pytest.mark.parametrize("dimension", [1024, 16384])
def test_v2_projection_bitwise_matches_pinned_official_reference(
    dimension: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "upstream/SubLoRA-bounds-for-LLMs"))
    from sublora.nn.projectors import RoundedDoubleKronQR as OfficialRoundedDoubleKronQR

    coordinate = torch.linspace(-1.0, 1.0, dimension)
    official = OfficialRoundedDoubleKronQR(
        1_182_720, dimension, [], [], seed=137
    ) @ coordinate
    adapted = RoundedDoubleKronQR(1_182_720, dimension, seed=137)(coordinate)

    assert torch.equal(adapted, official)


def test_v2_structure_cost_changes_s4k_length_without_mutating_v1() -> None:
    v1_manifest = json.loads(
        (ROOT / "runs/s4k_formal_v1/quantized_q11/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    counts = np.asarray(v1_manifest["quantization"]["counts"], dtype=np.uint32)
    v1 = description_length_upper_bound(
        counts, dimension=4096, structure_choice_bits=0
    )
    v2 = description_length_upper_bound(
        counts, dimension=4096, structure_choice_bits=2
    )

    assert v1["K_VLM_upper_bits"] == 11089
    assert v2["structure_choice_bits"] == 2
    assert v2["C_VLM_raw_upper_bits"] == v1["C_VLM_raw_upper_bits"] + 2
    assert v2["K_VLM_upper_bits"] == 11091


def test_v2_checkpoint_contract_binds_configuration_and_dimension(tmp_path: Path) -> None:
    registry = load_v2_subspace_registry(REGISTRY_PATH)
    contract = build_training_contract(
        registry_path=REGISTRY_PATH,
        registry=registry,
        configuration_id="S-1K",
        intrinsic_dimension=1024,
        world_size=2,
        dataset_size=1_147_346,
        steps_per_epoch=35_854,
    )
    checkpoint = {
        "progress": {
            "epoch": 2,
            "next_batch_in_epoch": 0,
            "global_step": 71_708,
            "total_steps": 71_708,
        },
        "subspace_params": torch.zeros(1024, dtype=torch.float32),
        "contract": contract,
    }
    checkpoint_path = tmp_path / "final.pt"
    checkpoint_path.write_bytes(b"presence-only")
    validate_v2_final_checkpoint(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        registry_path=REGISTRY_PATH,
        registry=registry,
        configuration_id="S-1K",
        intrinsic_dimension=1024,
    )

    changed = copy.deepcopy(checkpoint)
    changed["contract"]["configuration_id"] = "S-16K"
    with pytest.raises(RuntimeError, match="configuration_id"):
        validate_v2_final_checkpoint(
            checkpoint=changed,
            checkpoint_path=checkpoint_path,
            registry_path=REGISTRY_PATH,
            registry=registry,
            configuration_id="S-1K",
            intrinsic_dimension=1024,
        )


def test_v2_training_and_quantization_sources_do_not_name_locked_test_manifest() -> None:
    for path in (ROOT / "train_subspace_v2.py", ROOT / "quantize_final_subspace_v2.py"):
        source = path.read_text(encoding="utf-8")
        assert "dataset/locked_test" not in source
        assert "test_clusters.jsonl" not in source
