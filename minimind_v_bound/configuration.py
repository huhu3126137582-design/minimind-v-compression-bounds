from __future__ import annotations

import math
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


V2_REGISTRY_STATUS = "prospectively_frozen_extension_after_v1_test_unlock"
V2_FREEZE_MANIFEST = Path("configs/v2_subspace_protocol_freeze_manifest.json")


@dataclass(frozen=True)
class SubspaceCandidate:
    configuration_id: str
    intrinsic_dimension: int
    quantization_levels: int
    action: str
    output_directory: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_v2_protocol_freeze(*, root: Path, registry_path: Path) -> dict[str, Any]:
    manifest_path = root / V2_FREEZE_MANIFEST
    if not manifest_path.is_file():
        raise RuntimeError("v2 protocol freeze manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_v2_formal_model_training":
        raise RuntimeError("v2 protocol is not in its frozen pre-training state")
    expected_registry = manifest.get("artifact_sha256", {}).get(
        "configs/experiment_registry_v2_subspace.yaml"
    )
    if registry_path != (root / "configs/experiment_registry_v2_subspace.yaml").resolve():
        raise RuntimeError("formal v2 execution requires the frozen canonical registry")
    if _sha256_file(registry_path) != expected_registry:
        raise RuntimeError("v2 registry changed after protocol freezing")
    for relative_path, expected_hash in manifest.get("artifact_sha256", {}).items():
        path = root / relative_path
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise RuntimeError(f"v2 frozen protocol artifact changed: {relative_path}")
    if manifest.get("old_v1_test_policy", {}).get("blind_for_v2") is not False:
        raise RuntimeError("v2 freeze manifest must disclose that v1 test is no longer blind")
    return manifest


def load_v2_subspace_registry(path: Path) -> dict[str, Any]:
    registry = yaml.safe_load(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ValueError("unsupported v2 registry schema")
    if registry.get("registry_status") != V2_REGISTRY_STATUS:
        raise ValueError("v2 registry does not disclose its post-v1-test timing")
    candidates = registry.get("model_family", {}).get("candidates", [])
    identifiers = [candidate.get("configuration_id") for candidate in candidates]
    dimensions = [candidate.get("intrinsic_dimension") for candidate in candidates]
    if identifiers != ["S-1K", "S-4K", "S-16K"]:
        raise ValueError("v2 candidate order or identifiers changed")
    if dimensions != [1024, 4096, 16384]:
        raise ValueError("v2 intrinsic dimension family changed")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("v2 configuration identifiers are not unique")
    expected_structure_bits = math.ceil(math.log2(len(candidates)))
    description = registry.get("description_length", {})
    if description.get("structure_choice_bits") != expected_structure_bits:
        raise ValueError("v2 structure choice cost does not encode all candidates")
    certificate = registry.get("certificate", {})
    if certificate.get("model_count") != len(candidates):
        raise ValueError("v2 model count does not match candidate count")
    if not certificate.get("shared_sample_required"):
        raise ValueError("v2 requires one shared certificate sample")
    if certificate.get("reuse_v1_certificate_sample") is not False:
        raise ValueError("v2 must draw a fresh certificate sample")
    scope = registry.get("scope_and_timing", {})
    if scope.get("v1_locked_test_was_observed_before_this_registry") is not True:
        raise ValueError("v2 timing disclosure is missing")
    if registry.get("evaluation", {}).get(
        "old_v1_test_results_must_not_be_read_by_v2_training_or_model_selection_code"
    ) is not True:
        raise ValueError("v2 test-leakage prohibition is missing")
    return registry


def resolve_subspace_candidate(
    registry: dict[str, Any], configuration_id: str, *, root: Path
) -> SubspaceCandidate:
    matches = [
        candidate
        for candidate in registry["model_family"]["candidates"]
        if candidate["configuration_id"] == configuration_id
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate v2 configuration: {configuration_id}")
    candidate = matches[0]
    levels = int(candidate["quantization_levels"])
    if levels != int(registry["quantization"]["levels"]):
        raise ValueError("candidate and family quantization levels disagree")
    dimension = int(candidate["intrinsic_dimension"])
    if math.isqrt(dimension) ** 2 != dimension:
        raise ValueError("v2 RDKronQR dimensions must be perfect squares")
    output_directory = (root / candidate["output_directory"]).resolve()
    if root.resolve() not in output_directory.parents:
        raise ValueError("candidate output directory escapes the experiment root")
    return SubspaceCandidate(
        configuration_id=configuration_id,
        intrinsic_dimension=dimension,
        quantization_levels=levels,
        action=str(candidate["action"]),
        output_directory=output_directory,
    )
