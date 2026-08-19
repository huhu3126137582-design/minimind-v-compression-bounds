# MiniMind-V Compression-Bound Experiments

This repository contains the source, frozen contracts, tests, and compact result manifests for the MiniMind-V projector compression-bound study. The project is organized into two phases:

1. **Phase 1 / Milestone 1:** train exactly one `S-4K` Projector Subspace model and complete the minimum train, quantize, certificate, and held-out evaluation loop.
2. **Phase 2 / Milestone 6:** compare Projector Subspace and Projector-SubLoRA by selecting one lowest-bound representative from each method and checking whether bound ordering predicts raw held-out validation BPD ordering.

The current Phase 2 scope is v6 Q=11-only with nine frozen candidates. Its Q=11 scope was revised after formal training had started, so that comparison is post hoc rather than a prospective confirmation. Checkpoints, downloaded models, raw data, caches, and large run logs are intentionally excluded from Git.

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

## Phase 1: S-4K minimum closed loop

Phase 1 trains only the single configuration `S-4K`:

```text
parameterization: Projector Subspace
intrinsic dimension: d_P = 4096
quantization: Q_P = 11
model count: 1
```

The Phase 1 contract is `configs/experiment_registry.yaml`. Its reproducible artifacts are under `runs/s4k_formal_v1/`, including the final checkpoint, quantized artifact, shared certificate sample, certificate manifest, independent verification report, and fixed held-out evaluation manifest. The Phase 1 scripts are:

```text
train_s4k.py
quantize_final_s4k.py
verify_quantized_s4k.py
draw_certificate_sample.py
evaluate_certificate_s4k.py
verify_certificate_sample.py
verify_certificate_result.py
evaluate_locked_test_s4k.py
verify_locked_test_result.py
```

Milestone 1 answers whether this single compression configuration can complete the full protocol. The frozen certificate bound is `5.798642` bits/token with `alpha=0.125`, and the fixed held-out evaluation reports raw empirical risk `4.223987` bits/token. This is a closed-loop validation of the S-4K implementation, not a comparison between compression methods.

## Protocol checks and experiments

```bash
python verify_v6_compressibility_protocol.py
pytest -q tests/test_v6_compressibility.py
```

The Phase 2 Q=11 training and quantization entry points are `train_compressibility_v6.py`, `finalize_quantized_v6.py`, and `verify_quantized_v6.py`. Certification uses `draw_shared_certificate_sample_v6.py` followed by `evaluate_certificate_v6.py`; held-out evaluation uses `evaluate_heldout_validation_v6.py`. Read `方案.md` and `configs/experiment_registry_v6_compressibility.yaml` for exact seeds, split rules, optimizer schedule, and arguments.

### Milestone 5: non-vacuous certificate validation

Milestone 5 uses the frozen training-side certificate for every one of the nine candidates to verify whether the bound is non-vacuous:

```text
B_j < log2(V)
```

It does not read held-out validation outputs and does not use validation to retrain, requantize, choose `alpha`, remove candidates, or change the candidate set. Its only purpose is to establish that the fixed candidate family produces non-empty generalization certificates.

### Phase 2 / Milestone 6: prediction across compression methods

After Milestone 5 is frozen, Milestone 6 selects the smallest-bound representative separately within Projector Subspace and Projector-SubLoRA. Only those two representatives are then evaluated on held-out validation clusters. The main question is whether the training-side bound ordering predicts the relative raw validation BPD ordering between the two compression methods. This is a descriptive post hoc Bound–Validation comparison, not validation-based model selection and not a proof that validation error is mathematically controlled.

The completed M6 representatives and metrics are recorded in `runs/v6_compressibility/m6_bound_validation_report.json`:

| Method | Representative | Final bound | Raw validation BPD |
| --- | --- | ---: | ---: |
| Projector Subspace | `S-D256-Q11` | 5.037774 | 4.144036 |
| Projector-SubLoRA | `SL-R1-D256-Q11` | 5.019882 | 4.113346 |

All nine candidates are non-vacuous because their bounds are below `log2(6400) = 12.643856`. Projector-SubLoRA has both the lower bound and the lower raw validation BPD in this representative comparison, so the two directions agree. The held-out set had been opened during an earlier S-4K experiment; this result must therefore be reported as post hoc and descriptive.

Held-out analysis is post hoc and must not be presented as prospective model selection. Reproduction requires the same frozen inputs and substantial disk space.
