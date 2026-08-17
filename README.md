# MiniMind-V Compression-Bound Experiments

This repository contains the source, frozen contracts, and tests for the MiniMind-V projector compression-bound study. The published v6 scope is Q=11-only: nine pre-registered candidates (subspace and SubLoRA, dimensions 256/512/1024) are trained, quantized, certified, and optionally evaluated on held-out clusters. Checkpoints, downloaded models, raw data, caches, and run logs are intentionally excluded from Git.

## Reproducible setup

```bash
git clone --recurse-submodules https://github.com/huhu3126137582-design/minimind-v-compression-bounds.git
cd minimind-v-compression-bounds
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-experiment.txt
pip install -e .
pytest -q
```

The tested environment is Python 3.11, PyTorch 2.7.1, CUDA 12.8, and two GPUs.

## Required external artifacts

Place the frozen models and dataset at the paths documented by `configs/frozen_manifest.json` and `configs/v6_q11_only_protocol_revision_manifest.json`:

```text
model/llm_768.pth
model/siglip2-base-p32-256-ve/
dataset/raw/pretrain_i2t.parquet
dataset/clusters/image_clusters.sqlite3
```

These large files are not included. Verify their SHA-256 values before running experiments.

## Protocol checks and experiments

```bash
python verify_v6_compressibility_protocol.py
pytest -q tests/test_v6_compressibility.py
```

The Q=11 training and quantization entry points are `train_compressibility_v6.py`, `finalize_quantized_v6.py`, and `verify_quantized_v6.py`. Certification uses `draw_shared_certificate_sample_v6.py` followed by `evaluate_certificate_v6.py`; held-out evaluation uses `evaluate_heldout_validation_v6.py`. Read `方案.md` and `configs/experiment_registry_v6_compressibility.yaml` for exact seeds, split rules, optimizer schedule, and arguments.

Held-out analysis is post hoc and must not be presented as prospective model selection. Reproduction requires the same frozen inputs and substantial disk space.
