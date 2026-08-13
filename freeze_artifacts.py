from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import datasets
import numpy
import pyarrow
import torch
import transformers
import yaml


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_named_files(root: Path, relative_paths: list[str]) -> str:
    digest = hashlib.sha256()
    for relative_path in sorted(relative_paths):
        encoded_name = relative_path.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        path = root / relative_path
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record immutable experiment artifacts")
    parser.add_argument("--registry", type=Path, default=Path("configs/experiment_registry.yaml"))
    parser.add_argument("--output", type=Path, default=Path("configs/frozen_manifest.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {args.output}")

    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
    root = args.registry.resolve().parent.parent
    minimind_path = root / "upstream/minimind-v"
    sublora_path = root / "upstream/SubLoRA-bounds-for-LLMs"
    minimind_commit = git_commit(minimind_path)
    sublora_commit = git_commit(sublora_path)
    assert minimind_commit == registry["upstream"]["minimind_v"]["commit"]
    assert sublora_commit == registry["upstream"]["sublora_bounds"]["commit"]

    vision_path = root / registry["frozen_base"]["vision"]["local_path"]
    llm_path = root / registry["frozen_base"]["language"]["local_path"]
    tokenizer_path = root / registry["frozen_base"]["tokenizer"]["local_path"]
    dataset_path = root / registry["data"]["local_path"]
    required_paths = [
        vision_path / "model.safetensors",
        vision_path / "config.json",
        vision_path / "preprocessor_config.json",
        llm_path,
        tokenizer_path / "tokenizer.json",
        tokenizer_path / "tokenizer_config.json",
        dataset_path,
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required artifacts are missing: {missing}")

    manifest = {
        "schema_version": 1,
        "status": "artifacts_frozen_before_dataset_content_access",
        "experiment_registry_sha256": sha256_file(args.registry),
        "minimind_v_git_commit": minimind_commit,
        "sublora_bounds_git_commit": sublora_commit,
        "vision_checkpoint_sha256": sha256_file(vision_path / "model.safetensors"),
        "vision_config_sha256": sha256_file(vision_path / "config.json"),
        "vision_processor_config_sha256": sha256_file(
            vision_path / "preprocessor_config.json"
        ),
        "llm_checkpoint_sha256": sha256_file(llm_path),
        "tokenizer_sha256": sha256_named_files(
            tokenizer_path, ["tokenizer.json", "tokenizer_config.json"]
        ),
        "dataset_sha256": sha256_file(dataset_path),
        "projector_init_seed": registry["frozen_base"]["projector"]["initialization_seed"],
        "subspace_seed": registry["model"]["structured_projection"]["seed"],
        "parameter_name_and_shape_list": [
            ["mlp.0.weight", [768]],
            ["mlp.0.bias", [768]],
            ["mlp.1.weight", [768, 768]],
            ["mlp.1.bias", [768]],
            ["mlp.3.weight", [768, 768]],
            ["mlp.3.bias", [768]],
        ],
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "pyarrow": pyarrow.__version__,
            "numpy": numpy.__version__,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
