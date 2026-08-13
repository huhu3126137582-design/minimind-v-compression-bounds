from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from minimind_v_bound.compression.artifact import (
    canonical_compressed_model_sha256,
    canonical_json_bytes,
    load_and_verify_quantized_arrays,
    sha256_file,
    write_quantized_arrays,
)
from minimind_v_bound.compression.description_length import (
    DESCRIPTION_LENGTH_VERSION,
    description_length_upper_bound,
)
from minimind_v_bound.compression.quantization import QUANTIZATION_ALGORITHM_VERSION
from minimind_v_bound.configuration import (
    load_v2_subspace_registry,
    resolve_subspace_candidate,
    verify_v2_protocol_freeze,
)


ROOT = Path(__file__).resolve().parent
IMPORTER_VERSION = "frozen-s4k-v1-to-v2-description-family-v1"


def main() -> None:
    registry_path = ROOT / "configs/experiment_registry_v2_subspace.yaml"
    verify_v2_protocol_freeze(root=ROOT, registry_path=registry_path.resolve())
    registry = load_v2_subspace_registry(registry_path)
    candidate = resolve_subspace_candidate(registry, "S-4K", root=ROOT)
    if candidate.action != "import_frozen_v1":
        raise RuntimeError("S-4K is not registered as an exact v1 import")
    source_directory = ROOT / "runs/s4k_formal_v1/quantized_q11"
    expected_source_hash = registry["frozen_inputs"]["existing_s4k_v1"][
        "quantized_manifest_sha256"
    ]
    if sha256_file(source_directory / "manifest.json") != expected_source_hash:
        raise RuntimeError("frozen S-4K v1 manifest differs from the v2 registry")
    source_manifest, quantized = load_and_verify_quantized_arrays(source_directory)
    if source_manifest["hashes"]["compressed_model_sha256"] != registry[
        "frozen_inputs"
    ]["existing_s4k_v1"]["compressed_model_sha256_v1"]:
        raise RuntimeError("frozen S-4K v1 compressed-model hash mismatch")
    if source_manifest["quantization"]["dimension"] != candidate.intrinsic_dimension:
        raise RuntimeError("frozen S-4K dimension mismatch")
    if source_manifest["quantization"]["levels"] != candidate.quantization_levels:
        raise RuntimeError("frozen S-4K quantization-level mismatch")

    structure_bits = registry["description_length"]["structure_choice_bits"]
    length = description_length_upper_bound(
        quantized.counts,
        dimension=candidate.intrinsic_dimension,
        codebook_bits_per_center=registry["description_length"][
            "codebook_bits_per_center"
        ],
        structure_choice_bits=structure_bits,
    )
    descriptor = {
        "schema_version": 1,
        "experiment_id": registry["experiment_id"],
        "configuration_id": candidate.configuration_id,
        "candidate_id_code_bits": structure_bits,
        "parameterization": registry["model_family"]["parameterization"],
        "intrinsic_dimension": candidate.intrinsic_dimension,
        "subspace_seed": registry["model_family"]["structured_projection"]["seed"],
        "levels": candidate.quantization_levels,
        "quantization_algorithm_version": QUANTIZATION_ALGORITHM_VERSION,
        "v2_registry_sha256": sha256_file(registry_path),
        "frozen_manifest_sha256": sha256_file(ROOT / "configs/frozen_manifest.json"),
        "projector_init_manifest_sha256": sha256_file(
            ROOT / "configs/projector_init_manifest.json"
        ),
        "implementation_contract_sha256": sha256_file(
            ROOT / registry["frozen_inputs"]["implementation_contract"]
        ),
        "imported_exact_v1_symbols_and_centers": True,
    }
    compressed_hash = canonical_compressed_model_sha256(
        centers=quantized.centers,
        assignments=quantized.assignments,
        descriptor=descriptor,
    )
    materialized_hash = source_manifest["hashes"][
        "materialized_projector_parameters_sha256"
    ]
    hypothesis_descriptor = {
        "frozen_manifest_sha256": descriptor["frozen_manifest_sha256"],
        "compressed_model_sha256": compressed_hash,
        "materialized_projector_parameters_sha256": materialized_hash,
    }
    hypothesis_hash = hashlib.sha256(
        canonical_json_bytes(hypothesis_descriptor)
    ).hexdigest()

    output_directory = candidate.output_directory / "quantized_q11"
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".s4k_v2_import.", dir=output_directory.parent)
    )
    try:
        file_hashes = write_quantized_arrays(temporary, quantized)
        manifest = {
            "schema_version": 1,
            "status": "v2_quantized_model_frozen_before_shared_certificate_sampling",
            "importer_version": IMPORTER_VERSION,
            "source": {
                "v1_quantized_manifest": str(
                    (source_directory / "manifest.json").relative_to(ROOT)
                ),
                "v1_quantized_manifest_sha256": expected_source_hash,
                "v1_formal_checkpoint_sha256": source_manifest["source"][
                    "formal_checkpoint_sha256"
                ],
                "v1_compressed_model_sha256": source_manifest["hashes"][
                    "compressed_model_sha256"
                ],
                "v1_reconstructed_coordinate_sha256": source_manifest["hashes"][
                    "reconstructed_coordinate_sha256"
                ],
                "materialized_model_changed": False,
                "only_v2_family_descriptor_and_structure_cost_added": True,
            },
            "quantization": source_manifest["quantization"],
            "description_length": {
                "version": DESCRIPTION_LENGTH_VERSION,
                **length,
            },
            "hashes": {
                "projector_initialization_sha256": source_manifest["hashes"][
                    "projector_initialization_sha256"
                ],
                "reconstructed_coordinate_sha256": source_manifest["hashes"][
                    "reconstructed_coordinate_sha256"
                ],
                "materialized_projector_parameters_sha256": materialized_hash,
                "compressed_model_sha256": compressed_hash,
                "quantized_hypothesis_sha256": hypothesis_hash,
            },
            "compressed_model_descriptor": descriptor,
            "files": file_hashes,
            "shared_certificate_sample_seed": None,
        }
        with (temporary / "manifest.json").open("x", encoding="utf-8", newline="\n") as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
        os.rename(temporary, output_directory)
        for artifact in output_directory.iterdir():
            artifact.chmod(0o444)
        output_directory.chmod(0o555)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "configuration_id": candidate.configuration_id,
                "K_VLM_upper_bits_v1": source_manifest["description_length"][
                    "K_VLM_upper_bits"
                ],
                "K_VLM_upper_bits_v2": length["K_VLM_upper_bits"],
                "materialized_projector_parameters_sha256": materialized_hash,
                "v2_compressed_model_sha256": compressed_hash,
                "output_directory": str(output_directory),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
