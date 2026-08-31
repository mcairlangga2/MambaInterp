# Running the experiments

Run all commands from the repository root. Each script supports `--help` and writes
only to its requested output directory.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The first run downloads `tiiuae/falcon-mamba-7b` from Hugging Face. To choose a model
cache location, add `--cache-dir /path/to/cache` to a command.

For optional 4-bit loading:

```bash
python -m pip install -e ".[quantization]"
```

Then add `--quantize` to the experiment command. Quantization requires CUDA and can
change numerical results, so report it in any paper or artifact metadata.

## 2. Recall evaluation

Full evaluation over both dataset variants and every included length:

```bash
python scripts/evaluate_recall.py \
  --device cuda:0 \
  --output-dir results/recall
```

Evaluate selected lengths or one relation variant:

```bash
python scripts/evaluate_recall.py \
  --lengths 8 16 32 \
  --variants same \
  --device cuda:0 \
  --output-dir results/recall_same
```

Useful options:

- `--variants different same`: dataset families to evaluate;
- `--lengths 8 16 32 64 128`: sequence lengths;
- `--limit N`: first `N` examples from each CSV, useful for smoke tests;
- `--force`: recompute even when a complete evaluated CSV exists;
- `--confidence 0.95`: confidence level for Wilson intervals;
- `--max-new-tokens 2`: number of generated completion tokens.

Outputs include per-dataset `*_evaluated.csv` files,
`overall_recall_summary.csv`, `position_recall_stats.csv`, and
`position_recall_wilson_bands.png`. Existing complete per-example files are reused.

## 3. Recurrent channel intervention

Run identification, baseline evaluation, targeted intervention, and matched-random
control:

```bash
python scripts/run_channel_intervention.py \
  --dataset data/recall/8_same_relation.csv \
  --tau 0.7 \
  --p 0.7 \
  --top-layers 3 \
  --token-indices 1 2 3 4 \
  --seed 42 \
  --device cuda:0 \
  --output-dir results/channel_l8
```

`--token-indices` are zero-based and the default reproduces the old Python slice
`1:5`. Identification selects a deterministic example whose `target_position` is 1.
For every layer and channel, it multiplies `A_t` over context tokens after excluding
the first context token and the final two query tokens. A state dimension counts as
retained when its product is above `tau`; the channel qualifies when the retained
fraction is above `p`.

To reuse an earlier identification and avoid that stage:

```bash
python scripts/run_channel_intervention.py \
  --channels-json results/channel_l8/identified_channels.json \
  --dataset data/recall/8_same_relation.csv \
  --device cuda:0 \
  --output-dir results/channel_l8_repeat
```

Outputs include three per-example CSVs, `identified_channels.json`, `summary.csv`,
position-wise statistics, a confidence-band plot, and `metadata.json` containing the
intervention definition and channel budget.

## 4. State initialization

Evaluate one or more layers:

```bash
python scripts/run_state_init.py \
  --dataset data/recall/8_same_relation.csv \
  --layers 31 \
  --init uniform \
  --seed 42 \
  --device cuda:0 \
  --output-dir results/state_init_layer31
```

Run the original all-layer sweep:

```bash
python scripts/run_state_init.py \
  --dataset data/recall/8_same_relation.csv \
  --layers all \
  --init uniform \
  --device cuda:0 \
  --output-dir results/state_init_all
```

Available initial states are `uniform` on `[0, 1)`, standard `normal`, and
`constant` (use `--value`). Initialization is applied only to the selected layer's
prefill state. The all-layer sweep performs a full evaluation per layer and can take
many hours; use `--limit` to validate the pipeline first.

Outputs include baseline and per-layer prediction CSVs, `summary.csv`,
`position_stats.csv`, `recall_by_position.png`, and `metadata.json`.

## 5. Uploading to GitHub

After updating `CITATION.cff` and the dataset provenance note:

```bash
git init
git add .
git commit -m "Initial public release"
git branch -M main
git remote add origin https://github.com/YOUR-ACCOUNT/YOUR-REPOSITORY.git
git push -u origin main
```

The generated `results/` tree and model caches are ignored. The packaged CSV files
are below GitHub's per-file size limit.
