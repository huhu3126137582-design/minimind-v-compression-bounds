from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


V3_REGISTRY_STATUS = "prospectively_frozen_sublora_extension_after_v2_certificates"
V3_FREEZE_MANIFEST = Path("configs/v3_sublora_protocol_freeze_manifest.json")


@dataclass(frozen=True)
class SubLoRACandidate:
    configuration_id: str
    rank: int
    intrinsic_dimension: int
    quantization_levels: int
    output_directory: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_v3_protocol_freeze(*, root: Path, registry_path: Path) -> dict[str, Any]:
    manifest_path = root / V3_FREEZE_MANIFEST
    if not manifest_path.is_file():
        raise RuntimeError("v3 protocol freeze manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_v3_formal_model_training":
        raise RuntimeError("v3 protocol is not in its frozen pre-training state")
    canonical_registry = (root / "configs/experiment_registry_v3_sublora.yaml").resolve()
    if registry_path != canonical_registry:
        raise RuntimeError("formal v3 execution requires the frozen canonical registry")
    for relative_path, expected_hash in manifest.get("artifact_sha256", {}).items():
        path = root / relative_path
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise RuntimeError(f"v3 frozen protocol artifact changed: {relative_path}")
    return manifest


def load_v3_sublora_registry(path: Path) -> dict[str, Any]:
    registry = yaml.safe_load(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ValueError("unsupported v3 registry schema")
    if registry.get("registry_status") != V3_REGISTRY_STATUS:
        raise ValueError("v3 registry timing disclosure is missing")

    candidates = registry.get("model_family", {}).get("candidates", [])
    identifiers = [candidate.get("configuration_id") for candidate in candidates]
    ranks = [candidate.get("lora_rank") for candidate in candidates]
    dimensions = [candidate.get("intrinsic_dimension") for candidate in candidates]
    if identifiers != ["SL4-1K", "SL4-4K", "SL8-1K", "SL8-4K"]:
        raise ValueError("v3 candidate order or identifiers changed")
    if ranks != [4, 4, 8, 8] or dimensions != [1024, 4096, 1024, 4096]:
        raise ValueError("v3 rank/dimension matrix changed")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("v3 configuration identifiers are not unique")

    comparison = registry.get("comparison_family", {})
    all_ids = comparison.get("candidate_ids", [])
    if all_ids != [
        "S-1K", "S-4K", "S-16K", "SL4-1K", "SL4-4K", "SL8-1K", "SL8-4K"
    ]:
        raise ValueError("v3 comparison family must contain the seven fixed models")
    expected_bits = math.ceil(math.log2(len(all_ids)))
    if registry.get("description_length", {}).get("structure_choice_bits") != expected_bits:
        raise ValueError("v3 structure choice cost does not encode all seven models")
    certificate = registry.get("certificate", {})
    if certificate.get("model_count") != len(all_ids):
        raise ValueError("v3 certificate model count must be seven")
    if certificate.get("reuse_any_existing_certificate_sample") is not False:
        raise ValueError("v3 must draw a fresh sample after all seven models are fixed")
    if not certificate.get("shared_sample_required"):
        raise ValueError("v3 requires a shared sample")

    parameterization = registry.get("model_family", {}).get("parameterization_contract", {})
    if parameterization.get("theorem_coordinate_map") != "phi_P(u)=phi_P0+P_P*u":
        raise ValueError("v3 theorem-side SubLoRA map is not fixed")
    if parameterization.get("projector_map") != "omega=omega_0+LoRA_P(phi_P(u))":
        raise ValueError("v3 nonlinear Projector map is not fixed")
    if parameterization.get("target_linear_names") != ["mlp.1", "mlp.3"]:
        raise ValueError("v3 SubLoRA targets changed")
    if parameterization.get("train_layernorm") or parameterization.get("train_bias"):
        raise ValueError("v3 LayerNorm and biases must remain fixed")

    evaluation = registry.get("evaluation", {})
    if evaluation.get("old_v1_test_access") != "prohibited":
        raise ValueError("v3 old-test prohibition is missing")
    if evaluation.get("v2_certificate_artifact_access_during_training") != "prohibited":
        raise ValueError("v3 training must not access v2 certificate artifacts")
    return registry


def resolve_sublora_candidate(
    registry: dict[str, Any], configuration_id: str, *, root: Path
) -> SubLoRACandidate:
    matches = [
        candidate
        for candidate in registry["model_family"]["candidates"]
        if candidate["configuration_id"] == configuration_id
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate v3 configuration: {configuration_id}")
    candidate = matches[0]
    levels = int(candidate["quantization_levels"])
    if levels != int(registry["quantization"]["levels"]):
        raise ValueError("candidate and family quantization levels disagree")
    dimension = int(candidate["intrinsic_dimension"])
    if math.isqrt(dimension) ** 2 != dimension:
        raise ValueError("v3 RDKronQR intrinsic dimensions must be perfect squares")
    rank = int(candidate["lora_rank"])
    if rank not in (4, 8):
        raise ValueError("v3 only permits rank 4 or 8")
    output_directory = (root / candidate["output_directory"]).resolve()
    if root.resolve() not in output_directory.parents:
        raise ValueError("candidate output directory escapes the experiment root")
    return SubLoRACandidate(
        configuration_id=configuration_id,
        rank=rank,
        intrinsic_dimension=dimension,
        quantization_levels=levels,
        output_directory=output_directory,
    )
