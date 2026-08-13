# MiniMind-V Projector Compression-Bound Experiment

This workspace implements the preregistered S-4K minimal experiment described in
`方案.md`. The frozen base, source revisions, data hash, split, and training choices
are recorded under `configs/` and `dataset/manifests/`.

## Environment

```bash
git clone --recurse-submodules \
  https://github.com/huhu3126137582-design/minimind-v-compression-bounds.git
cd minimind-v-compression-bounds
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-experiment.txt
pip install -e .
python -m pytest -q
```

The project-local environment uses Python 3.11, PyTorch 2.7.1 with CUDA 12.8,
and recognizes both RTX 5090 GPUs.

## Locked artifacts

- `configs/experiment_registry.yaml`: preregistered S-4K experiment contract.
- `configs/frozen_manifest.json`: hashes of the frozen source, base models,
  tokenizer, dataset, and software versions.
- `configs/projector_init_manifest.json`: canonical random Projector initialization.
- `configs/implementation_contract.json`: locked RDKronQR, prompt, label-mask,
  train-only data access, and DDP semantics.
- `dataset/manifests/split_manifest.json`: immutable split counts and hashes.
- `dataset/manifests/train_clusters.jsonl`: only split manifest permitted to training.
- `dataset/locked_test/test_clusters.jsonl`: final test manifest; do not read before
  the quantized model, description length, alpha, and certificate are final.

## Re-run the split audit

```bash
python audit_split.py \
  --database dataset/clusters/image_clusters.sqlite3 \
  --train dataset/manifests/train_clusters.jsonl \
  --test dataset/locked_test/test_clusters.jsonl \
  --manifest dataset/manifests/split_manifest.json
```

Do not run a training command from the pinned upstream MiniMind-V repository directly:
its dataset pipeline includes random prompt transformations and does not enforce the
locked image-cluster split required by this experiment.

## Verification completed before formal training

```bash
pytest -q
python smoke_test_s4k.py --device cuda:0
CUDA_LAUNCH_BLOCKING=1 NCCL_DEBUG=WARN \
  torchrun --standalone --nproc_per_node=2 smoke_test_s4k_ddp.py
```

DDP must use `init_sync=False`: both ranks reconstruct the same hashed frozen base
and zero intrinsic coordinate, while broadcasting the complete frozen VLM triggers
an illegal-memory-access failure in this machine's NCCL stack. The two-rank smoke
test verifies that synchronized gradients produce bitwise-identical coordinates.

## Formal S-4K training

```bash
NCCL_DEBUG=WARN torchrun --standalone --nproc_per_node=2 train_s4k.py \
  --output-dir runs/s4k_formal_v1 \
  --num-workers 2 \
  --log-interval 100 \
  --checkpoint-interval 1000
```

The formal command intentionally has no `--max-steps`. It runs 35,854 steps per
epoch and 71,708 steps total. `last.pt` is an atomic resume checkpoint; `final.pt`
is created only after both complete epochs and is the preregistered selected model.

## External artifacts

Large immutable inputs are intentionally not stored in Git:

- `model/llm_768.pth`
- `model/siglip2-base-p32-256-ve/`
- `dataset/raw/pretrain_i2t.parquet`
- `dataset/clusters/image_clusters.sqlite3`

Their required paths and SHA-256 hashes are recorded in
`configs/frozen_manifest.json`. Build the cluster database with
`build_image_clusters.py`, and validate all frozen inputs with
`freeze_artifacts.py` before training.

The generated file
`runs/s4k_formal_v1/locked_test_evaluation/correct_token_log_probabilities.npy`
is excluded because it exceeds GitHub's 100 MiB per-file limit. It is reproducible
from the committed contract and checkpoint by rerunning `evaluate_locked_test_s4k.py`.

## Resume the current S-16K run

The committed `runs/v2_subspace/s16k/training/last.pt` checkpoint contains step
66,000 of 71,708, including the intrinsic coordinate and AdamW state. Resume it on
two GPUs with:

```bash
torchrun --standalone --nproc_per_node=2 train_subspace_v2.py \
  --configuration-id S-16K \
  --resume runs/v2_subspace/s16k/training/last.pt
```
