# MambaCode

Reproducible experiments for studying structured recall in
[Falcon-Mamba-7B](https://huggingface.co/tiiuae/falcon-mamba-7b). The repository
contains three focused workflows:

1. recall evaluation across sequence lengths and relation variants;
2. causal ablation of recurrent transition channels, with a matched-random control;
3. causal initialization of the SSM state at selected layers.


```

## Quick start

Use Python 3.10+ in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Run a small recall evaluation:

```bash
python scripts/evaluate_recall.py \
  --lengths 8 \
  --variants same \
  --limit 20 \
  --device cuda:0 \
  --output-dir results/recall_smoke
```

Run the original-style targeted channel intervention:

```bash
python scripts/run_channel_intervention.py \
  --dataset data/recall/8_same_relation.csv \
  --device cuda:0 \
  --output-dir results/channel_l8
```

Initialize the recurrent state at layer 31 from a uniform distribution:

```bash
python scripts/run_state_init.py \
  --dataset data/recall/8_same_relation.csv \
  --layers 31 \
  --init uniform \
  --device cuda:0 \
  --output-dir results/state_init_layer31
```

See [RUNNING.md](RUNNING.md) for full experiments, output schemas, hardware notes,
and all options.

## Experimental definitions

Recall uses greedy decoding and preserves the original metric: an example is correct
when the expected completion occurs as a case-insensitive substring of the newly
generated text. Position-wise plots use 95% Wilson score intervals by default.

The channel experiment identifies high-retention channels from the product of the
discretized recurrent transition `A_t` across context tokens. It selects the three
layers with the most qualifying channels, sets those channels' `A_t` values to zero
at zero-based token indices 1–4, and compares the result with an equal-size random
channel intervention.

The state-initialization experiment replaces the zero SSM state only at prompt prefill.
Autoregressive decoding continues from the resulting cached state; it is not reset at
each generated token.

## Reproducibility notes

- Causal experiments intentionally use the Python slow recurrence and require
  `transformers==4.49.0`. This is pinned because the scripts modify model internals.
- Seeds are applied to Python, NumPy, and PyTorch. CUDA deterministic settings are
  enabled, though exact cross-device bitwise identity is not guaranteed.
- Full-precision Falcon-Mamba-7B requires a high-memory CUDA GPU. Add the optional
  4-bit dependency with `python -m pip install -e ".[quantization]"`, then pass
  `--quantize` if memory is limited.
- Dataset provenance was not recorded in the source directory. The included files are
  reproduced unchanged; document their generation/source before archival publication.

## Testing

```bash
python -m pip install -e ".[dev]"
pytest
ruff check src scripts tests
```

## Citation and license

Update the generic author entry in [CITATION.cff](CITATION.cff) before making a release.
The project is distributed under Apache-2.0. The patched Falcon-Mamba recurrence is
adapted from Hugging Face Transformers; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
