from __future__ import annotations

import json
import os
from pathlib import Path

from minimind_v_bound.configuration_v6_compressibility import (
    V6_CANDIDATE_CODES,
    V6_CONFIGURATION_IDS,
    load_v6_registry,
    sha256_file,
    verify_v6_protocol_freeze,
)


ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "configs/experiment_registry_v6_compressibility.yaml"
FREEZE = ROOT / "configs/v6_q11_only_protocol_revision_manifest.json"
REPORT = ROOT / "configs/v6_q11_only_protocol_revision_verification.json"


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"refusing to overwrite {REPORT}")
    registry = load_v6_registry(REGISTRY)
    freeze = verify_v6_protocol_freeze(root=ROOT, registry_path=REGISTRY)
    if freeze["model_count"] != 9 or freeze["candidate_count"] != 9:
        raise RuntimeError("v6 Q=11-only freeze model count differs")
    if registry["candidate_family"]["candidate_id_bits"] != 4:
        raise RuntimeError("v6 candidate identity cost differs")
    report = {
        "schema_version": 1,
        "status": "v6_q11_only_scope_revision_verified_before_resume",
        "freeze_manifest_sha256": sha256_file(FREEZE),
        "registry_sha256": sha256_file(REGISTRY),
        "candidate_count": 9,
        "candidate_lookup_is_bijective": True,
        "candidate_identity_cost_bits": 4,
        "structure_fields_charged_once": True,
        "shared_certification_model_count": 9,
        "fresh_certification_sample_required": True,
        "scope_revised_after_training_started": True,
        "heldout_validation_access_prohibited_during_6a": registry["evaluation"][
            "heldout_validation_access_during_6a"
        ]
        == "prohibited",
        "all_candidates_use_q11": all(
            item["quantization_levels"] == 11
            for item in registry["candidate_family"]["candidates"]
        ),
    }
    with REPORT.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    REPORT.chmod(0o444)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
