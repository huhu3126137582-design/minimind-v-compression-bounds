from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


V6_REGISTRY_STATUS = "posthoc_v6_fixed_after_v5_results_before_v6_training"
V6_FREEZE_MANIFEST = Path("configs/v6_q11_only_protocol_revision_manifest.json")
V6_CONFIGURATION_IDS = (
    "S-D256-Q11",
    "S-D512-Q11",
    "S-D1024-Q11",
    "SL-R1-D256-Q11",
    "SL-R1-D512-Q11",
    "SL-R1-D1024-Q11",
    "SL-R4-D256-Q11",
    "SL-R4-D512-Q11",
    "SL-R4-D1024-Q11",
)
V6_CANDIDATE_CODES = ("0000", "0001", "0010", "0011", "0101", "0111", "1001", "1011", "1101")


@dataclass(frozen=True)
class V6Candidate:
    candidate_id: str
    configuration_id: str
    parameterization: str
    rank: int | None
    intrinsic_dimension: int
    quantization_levels: int
    output_directory: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_v6_registry(path: Path) -> dict[str, Any]:
    registry = yaml.safe_load(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ValueError("unsupported v6 registry schema")
    if registry.get("registry_status") != V6_REGISTRY_STATUS:
        raise ValueError("v6 timing disclosure is missing")
    if registry.get("revision_status") != "q11_only_scope_reduction_after_v6_training_started":
        raise ValueError("v6 Q=11-only scope revision is missing")
    family = registry.get("candidate_family", {})
    candidates = family.get("candidates", [])
    identifiers = tuple(item.get("configuration_id") for item in candidates)
    codes = tuple(item.get("candidate_id") for item in candidates)
    if identifiers != V6_CONFIGURATION_IDS:
        raise ValueError("v6 candidate identifiers or order changed")
    if codes != V6_CANDIDATE_CODES:
        raise ValueError("v6 candidate IDs must be the fixed sparse Q=11 lookup table")
    if family.get("candidate_count") != 9 or family.get("candidate_id_bits") != 4:
        raise ValueError("v6 must encode exactly nine Q=11 candidates with four bits")
    if family.get("unused_codewords") != [
        "0100", "0110", "1000", "1010", "1100", "1110", "1111"
    ]:
        raise ValueError("v6 Q=17 and reserved codewords must remain unused")
    expected = []
    for dimension in (256, 512, 1024):
        expected.append(("subspace", None, dimension, 11))
    for rank in (1, 4):
        for dimension in (256, 512, 1024):
            expected.append(("sublora", rank, dimension, 11))
    actual = [
        (
            item.get("parameterization"),
            item.get("lora_rank"),
            item.get("intrinsic_dimension"),
            item.get("quantization_levels"),
        )
        for item in candidates
    ]
    if actual != expected:
        raise ValueError("v6 method/rank/dimension/level lookup table changed")
    common = family.get("common_side_information", {})
    if common.get("target_linear_names") != ["mlp.1", "mlp.3"]:
        raise ValueError("v6 SubLoRA targets changed")
    if common.get("structured_projection_algorithm_version") != (
        "pinned-sublora-c606ea6-order2-arbitrary-tail-v2"
    ):
        raise ValueError("v6 projection algorithm is not fixed")
    training = registry.get("training", {})
    if training.get("epochs") != 4:
        raise ValueError("v6 requires four epochs")
    if training.get("expected_total_steps") != 143_416:
        raise ValueError("v6 total step contract changed")
    qat = registry.get("qat", {})
    qat_steps = math.ceil(
        training["expected_total_steps"]
        * qat.get("final_fraction_numerator", 0)
        / qat.get("final_fraction_denominator", 1)
    )
    if qat_steps != 14_342 or qat.get("expected_qat_steps") != qat_steps:
        raise ValueError("v6 QAT step count changed")
    if qat.get("expected_qat_start_step") != training["expected_total_steps"] - qat_steps:
        raise ValueError("v6 QAT start step changed")
    if qat.get("center_optimizer_state_at_qat_boundary") != (
        "reset_to_empty_before_first_qat_update"
    ):
        raise ValueError("v6 center optimizer boundary rule changed")
    description = registry.get("description_length", {})
    if description.get("structure_choice_bits") != 4:
        raise ValueError("v6 candidate identity must cost four bits")
    if description.get("no_separate_method_rank_dimension_or_level_cost") is not True:
        raise ValueError("v6 structure fields would be double charged")
    certificate = registry.get("certificate", {})
    if certificate.get("model_count") != 9:
        raise ValueError("v6 simultaneous certification must use M=9")
    if certificate.get("reuse_any_existing_certificate_sample") is not False:
        raise ValueError("v6 requires a fresh certification sample")
    if registry.get("evaluation", {}).get("heldout_validation_access_during_6a") != "prohibited":
        raise ValueError("v6 6A validation-access prohibition is missing")
    return registry


def resolve_v6_candidate(
    registry: dict[str, Any], configuration_id: str, *, root: Path
) -> V6Candidate:
    matches = [
        item
        for item in registry["candidate_family"]["candidates"]
        if item["configuration_id"] == configuration_id
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate v6 candidate: {configuration_id}")
    item = matches[0]
    output_directory = (root / item["output_directory"]).resolve()
    if root.resolve() not in output_directory.parents:
        raise ValueError("v6 output directory escapes the experiment root")
    return V6Candidate(
        candidate_id=item["candidate_id"],
        configuration_id=item["configuration_id"],
        parameterization=item["parameterization"],
        rank=item["lora_rank"],
        intrinsic_dimension=int(item["intrinsic_dimension"]),
        quantization_levels=int(item["quantization_levels"]),
        output_directory=output_directory,
    )


def verify_v6_protocol_freeze(*, root: Path, registry_path: Path) -> dict[str, Any]:
    canonical = (root / "configs/experiment_registry_v6_compressibility.yaml").resolve()
    if registry_path.resolve() != canonical:
        raise RuntimeError("formal v6 execution requires the canonical registry")
    manifest_path = root / V6_FREEZE_MANIFEST
    if not manifest_path.is_file():
        raise RuntimeError("v6 protocol freeze manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "q11_only_scope_revision_frozen_before_resume":
        raise RuntimeError("v6 Q=11-only revision is not frozen before resume")
    for relative, expected in manifest.get("artifact_sha256", {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"v6 frozen artifact changed: {relative}")
    return manifest
