# Causal Dictionary Learning for Genomic Language Models

Code accompanying the manuscript **"Causal dictionary learning reveals and
validates transcription-factor binding features in genomic language models."**

This repository trains top-k sparse autoencoders (SAEs) on the hidden activations
of genomic language models (Nucleotide Transformer, DNABERT-2), validates the
learned features against transcription-factor (TF) binding using a
composition-matched statistical test, and causally tests whether individual
features are used by the model via feature ablation. It reproduces every
quantitative result, figure, and table in the paper.

---

## 1. System requirements

**Software.**
- Linux (tested on Ubuntu 22.04 / RHEL 8; cluster: Delta HPC, SLURM)
- Python 3.11
- CUDA 12.1 (GPU strongly recommended for activation harvesting and causal ablation)
- ~50 GB free disk for activations and intermediate arrays

**Hardware.**
- Activation harvesting and causal ablation: 1 GPU with ≥40 GB memory
  (tested on NVIDIA A40 and A100). CPU-only is possible but slow.
- SAE training: 1 GPU, ~10 min per layer.
- Statistical tests and figures: CPU only.

**Key Python dependencies** (exact versions in `requirements.txt`):
`torch==2.5.1+cu121`, `transformers==4.44.2`, `numpy`, `scipy`, `pyfaidx`,
`einops`, `matplotlib`.

Installation of dependencies takes ~10 minutes on a normal broadband connection.

---

## 2. Installation

```bash
git clone https://github.com/sarwanpasha/Genomic_SAE.git
cd genomic-sae-causal

python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

To verify the install:

```bash
python -c "import torch, transformers, scipy, pyfaidx; print('OK', torch.__version__)"
```

---

## 3. Data

All inputs are public. `bash data/download_data.sh` fetches everything into
`data/`:

| Resource | Source | File |
|---|---|---|
| hg38 reference | UCSC Genome Browser | `hg38.fa` |
| Candidate cis-regulatory elements | ENCODE SCREEN | `encode_ccres_hg38.bed` |
| Motif models (CORE vertebrates) | JASPAR 2024 | `jaspar2024_core_vertebrates.txt` |
| CTCF ChIP-seq IDR peaks (GM12878) | ENCODE | `ctcf_gm12878_idr.bed` |
| GATA1 ChIP-seq IDR peaks (K562) | ENCODE `ENCFF148JKK` | `gata1_k562_idr.bed` |
| REST ChIP-seq IDR peaks (K562) | ENCODE `ENCFF126YZM` | `rest_k562_idr.bed` |

Model checkpoints are downloaded automatically from the HuggingFace Hub on first
use (`InstaDeepAI/nucleotide-transformer-v2-500m-multi-species`,
`zhihan1996/DNABERT-2-117M`).

---

## 4. Reproducing the paper end-to-end

The full pipeline is orchestrated by `run_all.sh`. On a single A40 GPU it takes
roughly 6–8 hours end-to-end (dominated by activation harvesting and causal
ablation). Each stage can also be run independently; see below.

```bash
bash run_all.sh            # full pipeline, all models and TFs
```

### Stage-by-stage

All stages read configuration from environment variables, documented in
`configs/default.env`. The canonical settings are `MODEL∈{nt,dnabert2}`,
`LAYER` (NT: 14, DNABERT-2: 6), `TF∈{CTCF,GATA1,REST,SCRAMBLE,GATA1SCRAM}`.

```bash
# (1) Harvest per-token activations at four layers
MODEL=nt python src/harvest_activations.py

# (2) Train a top-k SAE per layer (k=32, 16x dictionary)
MODEL=nt LAYER=14 python src/train_sae.py

# (3) Build bound / unbound-motif (GC-matched) / background windows
NPER=5000 python src/build_binding_windows.py
NPER=5000 python src/build_scramble_windows.py       # negative controls

# (4) Compute SAE feature activations on labeled windows
MODEL=nt LAYER=14 TF=CTCF python src/harvest_binding.py

# (5) Composition-matched binding test (Mann-Whitney, GC-controlled)
MODEL=nt LAYER=14 TF=CTCF python src/test_binding.py

# (6) Causal feature ablation (single feature + controls)
MODEL=nt LAYER=14 TF=CTCF FEATURE=8087 python src/causal_patch.py

# (7) Population causal test (top-15 features vs random controls)
MODEL=nt LAYER=14 TF=CTCF TOPN_FEAT=15 python src/causal_population.py

# (8) Regenerate all figures and tables from result JSONs
python src/make_figures.py
```

SLURM batch templates for a cluster are in `configs/slurm/`.

---

## 5. Expected outputs

Running the pipeline reproduces the following headline numbers (also in
`results/RESULTS_SUMMARY.md`). Small numeric variation (±1 feature) across
hardware is expected due to nondeterministic GPU reductions; see Section 7.

**Causally validated binding features (of 15 tested; random controls 0/15):**

| TF | NT (layer 14) | DNABERT-2 (layer 6) |
|---|---|---|
| CTCF  | 7/15  | 10/15 |
| GATA1 | 9/15  | 13/15 |
| REST  | 7/15  | 14/15 |
| SCRAMBLE (neg.)   | 0 features | 0 features |
| GATA1-SCRAM (neg.) | 0 features | 0 features |

Example single-feature result (NT CTCF feature 8087):
`KL_bound/KL_unbound ≈ 1.81, AUC = 0.626, P = 1.7e-18`;
random-feature and motif-only controls at `AUC ≈ 0.51`.

A minimal smoke test that runs in ~5 minutes on CPU, using bundled example data
for one TF and a subset of windows, is provided:

```bash
bash tests/smoke_test.sh      # verifies the pipeline end-to-end on a small input
```

Expected smoke-test output is in `tests/expected_output.txt`.

---

## 6. Repository layout

```
src/                 all analysis code
  harvest_activations.py   extract per-token model activations
  model_loader.py          tokenizer + model loading (both architectures)
  model_loader_lm.py       LM-headed loading for logit readout
  train_sae.py             top-k sparse autoencoder training
  build_binding_windows.py bound/unbound/background window construction
  build_scramble_windows.py negative-control windows
  harvest_binding.py       SAE activations on labeled windows
  test_binding.py          composition-matched binding test
  causal_patch.py          single-feature causal ablation
  causal_population.py      population causal test + controls
  make_figures.py          figures and tables
configs/             environment and SLURM templates
data/                download script; data land here (git-ignored)
results/             result JSONs, summary, figures
tests/               smoke test and expected output
requirements.txt     pinned dependencies
run_all.sh           end-to-end driver
LICENSE              MIT
```

---

## 7. Reproducibility notes

- **Random seeds.** All stochastic steps (window sampling, SAE initialization,
  random-feature control selection) are seeded (`SEED`, default 0). Set it in
  `configs/default.env`.
- **Determinism.** GPU floating-point reductions are not bitwise-deterministic;
  the composition-matched and causal tests are robust to this, but exact
  per-feature AUCs may differ in the third decimal. Aggregate counts (X/15) are
  stable across seeds and hardware in our testing.
- **Model versions.** We pin `transformers==4.44.2`. DNABERT-2 requires an
  in-memory patch that disables its optional Flash-Attention path so it runs on
  standard GPUs; this is handled automatically in `model_loader_lm.py`.

---

## 9. License

Released under the MIT License (see `LICENSE`). All input data are subject to the
licenses of their respective providers (ENCODE, JASPAR, UCSC, HuggingFace model
authors).
