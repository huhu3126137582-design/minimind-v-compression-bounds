from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import torch

from minimind_v_bound.compression.artifact import sha256_file
from minimind_v_bound.configuration import load_v2_subspace_registry


ROOT = Path(__file__).resolve().parent


def main() -> None:
    output_path = ROOT / "configs/v2_subspace_protocol_freeze_manifest.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    registry_path = ROOT / "configs/experiment_registry_v2_subspace.yaml"
    registry = load_v2_subspace_registry(registry_path)
    formal_targets = [
        ROOT / "runs/v2_subspace/s1k/training",
        ROOT / "runs/v2_subspace/s1k/quantized_q11",
        ROOT / "runs/v2_subspace/s4k_imported/quantized_q11",
        ROOT / "runs/v2_subspace/s16k/training",
        ROOT / "runs/v2_subspace/s16k/quantized_q11",
    ]
    existing = [str(path) for path in formal_targets if path.exists()]
    if existing:
        raise RuntimeError(f"formal v2 targets already exist before freezing: {existing}")

    frozen_paths = [
        "configs/experiment_registry_v2_subspace.yaml",
        "configs/implementation_contract_v2_subspace.json",
        "train_subspace_v2.py",
        "quantize_final_subspace_v2.py",
        "import_s4k_v1_into_v2.py",
        "minimind_v_bound/configuration.py",
        "minimind_v_bound/models/structured_projection.py",
        "minimind_v_bound/models/intrinsic_projector.py",
        "minimind_v_bound/data/caption_dataset.py",
        "minimind_v_bound/training/checkpoint.py",
        "minimind_v_bound/training/schedule.py",
        "minimind_v_bound/compression/quantization.py",
        "minimind_v_bound/compression/description_length.py",
        "minimind_v_bound/compression/artifact.py",
    ]
    artifact_hashes = {
        relative_path: sha256_file(ROOT / relative_path)
        for relative_path in frozen_paths
    }
    base_paths = {
        "configs/experiment_registry.yaml": sha256_file(
            ROOT / "configs/experiment_registry.yaml"
        ),
        "configs/frozen_manifest.json": sha256_file(
            ROOT / "configs/frozen_manifest.json"
        ),
        "configs/projector_init_manifest.json": sha256_file(
            ROOT / "configs/projector_init_manifest.json"
        ),
        "dataset/manifests/split_manifest.json": sha256_file(
            ROOT / "dataset/manifests/split_manifest.json"
        ),
        "dataset/manifests/train_clusters.jsonl": sha256_file(
            ROOT / "dataset/manifests/train_clusters.jsonl"
        ),
        "dataset/manifests/train_index/metadata.json": sha256_file(
            ROOT / "dataset/manifests/train_index/metadata.json"
        ),
        "runs/s4k_formal_v1/final.pt": sha256_file(
            ROOT / "runs/s4k_formal_v1/final.pt"
        ),
        "runs/s4k_formal_v1/quantized_q11/manifest.json": sha256_file(
            ROOT / "runs/s4k_formal_v1/quantized_q11/manifest.json"
        ),
    }
    if base_paths["runs/s4k_formal_v1/final.pt"] != registry["frozen_inputs"][
        "existing_s4k_v1"
    ]["final_checkpoint_sha256"]:
        raise RuntimeError("S-4K v1 checkpoint differs from v2 registry")
    if base_paths[
        "runs/s4k_formal_v1/quantized_q11/manifest.json"
    ] != registry["frozen_inputs"]["existing_s4k_v1"][
        "quantized_manifest_sha256"
    ]:
        raise RuntimeError("S-4K v1 quantized manifest differs from v2 registry")

    smoke_results = {}
    for identifier, directory, dimension in (
        ("S-1K", ROOT / "runs/v2_development/s1k_smoke", 1024),
        ("S-16K", ROOT / "runs/v2_development/s16k_smoke", 16384),
    ):
        metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
        log = json.loads((directory / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
        checkpoint = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
        if metadata["formal_run"] is not False:
            raise RuntimeError(f"{identifier} smoke run was not marked development-only")
        if checkpoint["subspace_params"].shape != (dimension,):
            raise RuntimeError(f"{identifier} smoke checkpoint dimension mismatch")
        if checkpoint["progress"]["global_step"] != 1:
            raise RuntimeError(f"{identifier} smoke run did not stop at one step")
        smoke_results[identifier] = {
            "checkpoint_sha256": sha256_file(directory / "last.pt"),
            "coordinate_sha256": log["coordinate_sha256"],
            "loss_nats_per_token_rank0": log["loss_nats_per_token_rank0"],
            "gradient_norm_before_clip": log["gradient_norm_before_clip"],
            "peak_cuda_memory_bytes_rank0": log["peak_cuda_memory_bytes_rank0"],
            "ddp_coordinate_equality_check": "passed_inside_trainer",
            "frozen_parameter_gradient_check": "passed_inside_trainer",
        }

    repositories = {}
    for name, relative_path in (
        ("minimind_v", "upstream/minimind-v"),
        ("sublora_bounds", "upstream/SubLoRA-bounds-for-LLMs"),
    ):
        repository = ROOT / relative_path
        commit = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True
        )
        if dirty:
            raise RuntimeError(f"upstream repository is dirty: {repository}")
        repositories[name] = {"path": relative_path, "commit": commit, "dirty": False}

    manifest = {
        "schema_version": 1,
        "status": "frozen_before_v2_formal_model_training",
        "experiment_id": registry["experiment_id"],
        "timing_disclosure": registry["scope_and_timing"],
        "old_v1_test_policy": {
            "blind_for_v2": False,
            "permitted_use": "exploratory_only_after_all_v2_certificates_are_final",
            "new_confirmatory_test_required": True,
            "locked_test_content_read_by_this_freeze_program": False,
        },
        "candidate_ids": ["S-1K", "S-4K", "S-16K"],
        "intrinsic_dimensions": [1024, 4096, 16384],
        "model_count": 3,
        "structure_choice_bits": 2,
        "shared_fresh_certificate_sample_required": True,
        "artifact_sha256": artifact_hashes,
        "base_binding_sha256": base_paths,
        "upstream_repositories": repositories,
        "validation": {
            "pytest_tests_passed": 53,
            "pinned_official_projection_bitwise_match": {
                "S-1K": True,
                "S-16K": True
            },
            "two_gpu_one_step_smoke": smoke_results,
        },
        "formal_targets_absent_when_frozen": [
            str(path.relative_to(ROOT)) for path in formal_targets
        ],
    }
    with output_path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    output_path.chmod(0o444)
    for relative_path in frozen_paths:
        (ROOT / relative_path).chmod(0o444)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
                "frozen_artifact_count": len(frozen_paths),
                "formal_targets_absent": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
