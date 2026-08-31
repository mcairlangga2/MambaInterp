#!/usr/bin/env python3
"""Evaluate causal SSM state initialization at selected Falcon-Mamba layers."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mambacode.evaluation import (
    atomic_to_csv,
    atomic_to_json,
    evaluate_dataframe,
    load_recall_dataset,
    position_recall_stats,
)
from mambacode.interventions import (
    SlowForwardConfig,
    StateInitConfig,
    force_slow_path,
    get_mixers,
    patch_mixers,
)
from mambacode.modeling import DEFAULT_MODEL, load_model_and_tokenizer, seed_everything


plt.switch_backend("Agg")


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=repository / "data" / "recall" / "8_same_relation.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository / "results" / "state_init",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["all"],
        help="Layer indices, or 'all' (the original layer sweep).",
    )
    parser.add_argument("--init", choices=["uniform", "normal", "constant"], default="uniform")
    parser.add_argument("--value", type=float, default=0.0, help="Value used by constant init.")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--quantize", action="store_true")
    return parser.parse_args()


def resolve_layers(values: list[str], number_of_layers: int) -> list[int]:
    if "all" in values:
        if len(values) != 1:
            raise ValueError("Use either --layers all or explicit indices, not both.")
        return list(range(number_of_layers))
    layers = list(dict.fromkeys(int(value) for value in values))
    invalid = [layer for layer in layers if not 0 <= layer < number_of_layers]
    if invalid:
        raise ValueError(
            f"Invalid layers {invalid}; valid range is 0..{number_of_layers - 1}."
        )
    return layers


def save_plot(stats: pd.DataFrame, layers: list[int], path: Path) -> None:
    groups = [layers[index : index + 4] for index in range(0, len(layers), 4)]
    columns = min(4, max(1, len(groups)))
    rows = math.ceil(len(groups) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.2 * columns, 3.3 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()
    baseline = stats[stats["condition"].eq("baseline")].sort_values("target_position")
    for axis, group in zip(axes_flat, groups):
        axis.plot(
            baseline["target_position"],
            baseline["recall"],
            color="black",
            linestyle="--",
            linewidth=2,
            label="baseline",
        )
        for layer in group:
            subset = stats[stats["condition"].eq(f"layer_{layer}")].sort_values(
                "target_position"
            )
            axis.plot(
                subset["target_position"],
                subset["recall"],
                linewidth=1.5,
                label=f"layer {layer}",
            )
        axis.set_title(f"Layers {group[0]}–{group[-1]}")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    for axis in axes_flat[len(groups) :]:
        axis.axis("off")
    fig.supxlabel("Target position")
    fig.supylabel("Recall accuracy")
    fig.suptitle("Baseline vs. nonzero SSM state initialization", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    frame = load_recall_dataset(args.dataset.expanduser().resolve(), args.limit)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        args.device,
        args.cache_dir,
        args.quantize,
    )
    force_slow_path()
    layers = resolve_layers(args.layers, len(get_mixers(model)))

    results: dict[str, pd.DataFrame] = {}
    baseline = evaluate_dataframe(
        frame, model, tokenizer, args.device, args.max_new_tokens, "baseline"
    )
    results["baseline"] = baseline
    atomic_to_csv(baseline, output_dir / "baseline_per_example.csv")

    init_config = StateInitConfig(mode=args.init, value=args.value)
    for layer in layers:
        # Reset before each layer so the random intervention is reproducible and
        # paired across equally shaped layer states.
        seed_everything(args.seed)
        config = {layer: SlowForwardConfig(state_init=init_config)}
        with patch_mixers(model, config):
            evaluated = evaluate_dataframe(
                frame,
                model,
                tokenizer,
                args.device,
                args.max_new_tokens,
                f"state init layer {layer}",
            )
        condition = f"layer_{layer}"
        results[condition] = evaluated
        atomic_to_csv(evaluated, output_dir / f"{condition}_per_example.csv")

    stats_parts = []
    summary_rows = []
    for condition, result in results.items():
        stats = position_recall_stats(result, args.confidence)
        stats.insert(0, "condition", condition)
        stats_parts.append(stats)
        summary_rows.append(
            {
                "condition": condition,
                "recall": float(result["is_correct"].mean()),
                "n_examples": len(result),
            }
        )
    all_stats = pd.concat(stats_parts, ignore_index=True)
    atomic_to_csv(all_stats, output_dir / "position_stats.csv")
    atomic_to_csv(pd.DataFrame(summary_rows), output_dir / "summary.csv")
    atomic_to_json(
        {
            "model": args.model,
            "dataset": str(args.dataset.expanduser().resolve()),
            "seed": args.seed,
            "initialization": args.init,
            "constant_value": args.value if args.init == "constant" else None,
            "layers": layers,
            "initialization_applied_during_prefill_only": True,
            "limit": args.limit,
        },
        output_dir / "metadata.json",
    )
    save_plot(all_stats, layers, output_dir / "recall_by_position.png")
    print(f"Saved state-initialization results to {output_dir}")


if __name__ == "__main__":
    main()
