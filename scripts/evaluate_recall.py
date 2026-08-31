#!/usr/bin/env python3
"""Evaluate recall accuracy across the packaged synthetic datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

from mambacode.evaluation import (
    atomic_to_csv,
    evaluate_dataframe,
    load_cached_evaluation,
    load_recall_dataset,
    position_recall_stats,
)
from mambacode.modeling import DEFAULT_MODEL, load_model_and_tokenizer, seed_everything


plt.switch_backend("Agg")


VARIANTS = {
    "different": ("Different relations", "{length}.csv"),
    "same": ("Same relation", "{length}_same_relation.csv"),
}


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=repository / "data" / "recall")
    parser.add_argument("--output-dir", type=Path, default=repository / "results" / "recall")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--lengths", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Use the first N examples per dataset.")
    parser.add_argument("--quantize", action="store_true", help="Load the model in 4-bit.")
    parser.add_argument("--force", action="store_true", help="Ignore cached evaluated CSVs.")
    return parser.parse_args()


def save_plot(stats: pd.DataFrame, path: Path, confidence: float, variants: list[str]) -> None:
    fig, axes = plt.subplots(
        1,
        len(variants),
        figsize=(7 * len(variants), 4.8),
        sharey=True,
        squeeze=False,
    )
    for axis, variant in zip(axes[0], variants):
        relation_type = VARIANTS[variant][0]
        subset = stats[stats["relation_type"].eq(relation_type)]
        for length in sorted(subset["sequence_length"].unique()):
            length_stats = subset[subset["sequence_length"].eq(length)].sort_values(
                "target_position"
            )
            line = axis.plot(
                length_stats["target_position"],
                length_stats["recall"],
                linewidth=1.8,
                label=(
                    f"L={length} "
                    f"(overall={length_stats['overall_accuracy'].iloc[0]:.3f})"
                ),
            )[0]
            axis.fill_between(
                length_stats["target_position"],
                length_stats["ci_lower"],
                length_stats["ci_upper"],
                color=line.get_color(),
                alpha=0.16,
                linewidth=0,
            )
        axis.set_title(relation_type)
        axis.set_xlabel("Target position")
        axis.set_ylim(-0.02, 1.02)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, fontsize=9)
    axes[0, 0].set_ylabel("Recall accuracy")
    fig.suptitle(f"Position-wise recall with {confidence:.0%} Wilson intervals")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    evaluated_by_key: dict[tuple[str, int], pd.DataFrame] = {}
    for variant in dict.fromkeys(args.variants):
        relation_type, template = VARIANTS[variant]
        for length in dict.fromkeys(args.lengths):
            source = args.data_dir / template.format(length=length)
            frame = load_recall_dataset(source, args.limit)
            result = output_dir / f"{source.stem}_evaluated.csv"
            cached = None if args.force else load_cached_evaluation(result, len(frame))
            job = (variant, relation_type, length, source, result, frame)
            jobs.append(job)
            if cached is not None:
                print(f"CACHE {result}")
                evaluated_by_key[(variant, length)] = cached

    pending = [job for job in jobs if (job[0], job[2]) not in evaluated_by_key]
    if pending:
        model, tokenizer = load_model_and_tokenizer(
            args.model,
            args.device,
            args.cache_dir,
            args.quantize,
        )
        for variant, _, length, source, result, frame in pending:
            print(f"RUN   {source} -> {result}")
            evaluated = evaluate_dataframe(
                frame,
                model,
                tokenizer,
                args.device,
                args.max_new_tokens,
                source.stem,
            )
            atomic_to_csv(evaluated, result)
            evaluated_by_key[(variant, length)] = evaluated
    else:
        print("All requested evaluations were loaded from cache.")

    stats_parts = []
    summary_rows = []
    for variant, relation_type, length, _, _, _ in jobs:
        evaluated = evaluated_by_key[(variant, length)]
        stats = position_recall_stats(evaluated, args.confidence)
        stats.insert(0, "sequence_length", length)
        stats.insert(0, "relation_type", relation_type)
        accuracy = float(evaluated["is_correct"].mean())
        stats["overall_accuracy"] = accuracy
        stats_parts.append(stats)
        summary_rows.append(
            {
                "relation_type": relation_type,
                "sequence_length": length,
                "overall_accuracy": accuracy,
                "n_examples": len(evaluated),
            }
        )

    all_stats = pd.concat(stats_parts, ignore_index=True)
    atomic_to_csv(all_stats, output_dir / "position_recall_stats.csv")
    atomic_to_csv(pd.DataFrame(summary_rows), output_dir / "overall_recall_summary.csv")
    save_plot(
        all_stats,
        output_dir / "position_recall_wilson_bands.png",
        args.confidence,
        list(dict.fromkeys(args.variants)),
    )
    print(f"Saved recall results to {output_dir}")


if __name__ == "__main__":
    main()
