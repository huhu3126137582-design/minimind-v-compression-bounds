#!/usr/bin/env python3
"""Persistent, fail-closed supervisor for frozen Milestone 6A candidates.

The supervisor only orchestrates the already-frozen training, finalization, and
independent-verification programs.  It deliberately stops before fresh shared
certification sampling and never imports or reads held-out validation data.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from minimind_v_bound.configuration_v6_compressibility import (
    load_v6_registry,
    resolve_v6_candidate,
)
from verify_v6_compressibility_protocol import verify_v6_protocol_freeze


ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "configs/experiment_registry_v6_compressibility.yaml"
RUN_ROOT = ROOT / "runs/v6_compressibility"
STATE_PATH = RUN_ROOT / "supervisor_state.json"
EVENT_PATH = RUN_ROOT / "supervisor_events.jsonl"
LOCK_PATH = RUN_ROOT / "supervisor.lock"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all frozen v6 candidates through train/finalize/verify"
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--status", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def record(event: str, **fields: Any) -> None:
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "event": event,
        **fields,
    }
    with EVENT_PATH.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    atomic_json(STATE_PATH, payload)


def matching_training_processes(configuration_id: str) -> list[int]:
    matches: list[int] = []
    own_pid = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (
            "train_compressibility_v6.py" in command
            and f"--configuration-id {configuration_id}" in command
        ):
            matches.append(int(entry.name))
    return sorted(matches)


def run_logged(command: list[str], log_path: Path, environment: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] COMMAND {json.dumps(command)}\n")
        output.flush()
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
        output.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] EXIT {result.returncode}\n")
        output.flush()
        os.fsync(output.fileno())
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {command}")


def validate_existing_verification(path: Path, configuration_id: str) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "v6_qat_hypothesis_independently_verified_before_sampling":
        raise RuntimeError(f"unexpected verification status in {path}")
    if report.get("configuration_id") != configuration_id:
        raise RuntimeError(f"verification configuration mismatch in {path}")
    if report.get("fresh_certification_sample_seed") is not None:
        raise RuntimeError(f"certification sampling occurred prematurely in {path}")
    if report.get("heldout_validation_accessed") is not False:
        raise RuntimeError(f"held-out validation access flag is not false in {path}")


def train_candidate(configuration_id: str, training_directory: Path, poll_seconds: int) -> None:
    final_path = training_directory / "final.pt"
    if final_path.exists():
        record("training_already_complete", configuration_id=configuration_id)
        return

    processes = matching_training_processes(configuration_id)
    if processes:
        record(
            "adopting_existing_training",
            configuration_id=configuration_id,
            pids=processes,
        )
        while not final_path.exists():
            time.sleep(poll_seconds)
            processes = matching_training_processes(configuration_id)
            if not processes and not final_path.exists():
                last_path = training_directory / "last.pt"
                if not last_path.exists():
                    raise RuntimeError(
                        f"adopted training {configuration_id} exited without final.pt or last.pt"
                    )
                record(
                    "adopted_training_interrupted_resuming",
                    configuration_id=configuration_id,
                    checkpoint=str(last_path.relative_to(ROOT)),
                )
                break
        if final_path.exists():
            while matching_training_processes(configuration_id):
                time.sleep(2)
            record("training_complete", configuration_id=configuration_id)
            return

    last_path = training_directory / "last.pt"
    if training_directory.exists() and any(training_directory.iterdir()) and not last_path.exists():
        raise RuntimeError(
            f"non-empty training directory for {configuration_id} has no resumable last.pt"
        )

    command = [
        str(ROOT / ".venv/bin/torchrun"),
        "--standalone",
        "--nproc_per_node=2",
        str(ROOT / "train_compressibility_v6.py"),
        "--configuration-id",
        configuration_id,
        "--num-workers",
        "2",
        "--log-interval",
        "100",
        "--checkpoint-interval",
        "1000",
    ]
    if last_path.exists():
        command.extend(["--resume", str(last_path)])
        event = "training_resume_started"
    else:
        event = "training_started"
    record(event, configuration_id=configuration_id)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0,1"
    run_logged(
        command,
        RUN_ROOT / f"{configuration_id.lower().replace('-', '_')}_formal_training_stdout.log",
        environment,
    )
    if not final_path.exists():
        raise RuntimeError(f"training exited successfully without final.pt: {configuration_id}")
    record("training_complete", configuration_id=configuration_id)


def process_candidate(candidate: Any, poll_seconds: int) -> None:
    configuration_id = candidate.configuration_id
    training_directory = candidate.output_directory / "training"
    train_candidate(configuration_id, training_directory, poll_seconds)

    artifact_directory = candidate.output_directory / f"quantized_q{candidate.quantization_levels}"
    finalization_log = candidate.output_directory / "finalization_stdout.log"
    if not artifact_directory.exists():
        record("finalization_started", configuration_id=configuration_id)
        run_logged(
            [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "finalize_quantized_v6.py"),
                "--configuration-id",
                configuration_id,
            ],
            finalization_log,
        )
        record("finalization_complete", configuration_id=configuration_id)
    elif not (artifact_directory / "manifest.json").is_file():
        raise RuntimeError(f"incomplete quantized artifact directory: {artifact_directory}")
    else:
        record("finalization_already_complete", configuration_id=configuration_id)

    verification_path = candidate.output_directory / (
        f"quantized_q{candidate.quantization_levels}_verification.json"
    )
    if not verification_path.exists():
        record("verification_started", configuration_id=configuration_id)
        run_logged(
            [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "verify_quantized_v6.py"),
                "--configuration-id",
                configuration_id,
            ],
            candidate.output_directory / "verification_stdout.log",
        )
        record("verification_complete", configuration_id=configuration_id)
    validate_existing_verification(verification_path, configuration_id)
    record("candidate_complete", configuration_id=configuration_id)


def main() -> None:
    args = parse_args()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if args.status:
        if STATE_PATH.exists():
            print(STATE_PATH.read_text(encoding="utf-8"), end="")
        else:
            print(json.dumps({"status": "not_started"}, indent=2))
        return
    if args.poll_seconds < 5:
        raise ValueError("poll interval must be at least five seconds")

    lock_handle = LOCK_PATH.open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another v6 supervisor already owns the lock") from exc

    try:
        verification = verify_v6_protocol_freeze(root=ROOT, registry_path=REGISTRY)
        registry = load_v6_registry(REGISTRY)
        candidates = [
            resolve_v6_candidate(registry, item["configuration_id"], root=ROOT)
            for item in registry["candidate_family"]["candidates"]
        ]
        if len(candidates) != 9:
            raise RuntimeError("frozen v6 registry does not contain exactly nine Q=11 candidates")
        record(
            "supervisor_started",
            pid=os.getpid(),
            candidate_count=len(candidates),
            protocol_status=verification["status"],
            heldout_validation_accessed=False,
        )
        for index, candidate in enumerate(candidates, start=1):
            record(
                "candidate_entered",
                index=index,
                candidate_count=len(candidates),
                configuration_id=candidate.configuration_id,
                candidate_id=candidate.candidate_id,
            )
            process_candidate(candidate, args.poll_seconds)
        record(
            "all_candidates_verified",
            candidate_count=len(candidates),
            next_action="draw_one_fresh_shared_certification_sample",
            heldout_validation_accessed=False,
        )
    except BaseException as exc:
        record("supervisor_failed", error=repr(exc), heldout_validation_accessed=False)
        raise


if __name__ == "__main__":
    main()
