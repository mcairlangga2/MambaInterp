#!/usr/bin/env python3
"""Run baseline, targeted channel, and matched-random causal interventions."""

from __future__ import annotations

import argparse
import json
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
    channel_patch_configurations,
    deserialize_channel_map,
    force_slow_path,
    identify_retention_channels,
    patch_mixers,
    sample_random_channel_map,
    select_top_layers,
    serialize_channel_map,
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
        default=repository / "results" / "channel_intervention",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--channels-json", type=Path, help="Reuse previously identified channels.")
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--p", type=float, default=0.7)
    parser.add_argument("--top-layers", type=int, default=3)
    parser.add_argument(
        "--token-indices",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Zero-based context token indices whose recurrent A values are zeroed.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--quantize", action="store_true")
    return parser.parse_args()


def condition_stats(frame: pd.DataFrame, condition: str, confidence: float) -> pd.DataFrame:
    stats = position_recall_stats(frame, confidence)
    stats.insert(0, "condition", condition)
    return stats


def save_plot(stats: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    for condition, subset in stats.groupby("condition", sort=False):
        subset = subset.sort_values("target_position")
        line = axis.plot(
            subset["target_position"],
            subset["recall"],
            marker="o",
            linewidth=1.8,
            label=condition,
        )[0]
        axis.fill_between(
            subset["target_position"],
            subset["ci_lower"],
            subset["ci_upper"],
            color=line.get_color(),
            alpha=0.15,
            linewidth=0,
        )
    axis.set(xlabel="Target position", ylabel="Recall accuracy", ylim=(-0.02, 1.02))
    axis.set_title("Falcon-Mamba recurrent channel intervention")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
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

    if args.channels_json is not None:
        payload = json.loads(args.channels_json.read_text())
        channel_map = deserialize_channel_map(payload)
    else:
        candidates = frame[frame["target_position"].eq(1)]
        if candidates.empty:
            raise ValueError("Channel identification requires a target_position == 1 row.")
        prompt = str(candidates.sample(n=1, random_state=args.seed).iloc[0]["input"])
        channel_map = identify_retention_channels(
            model,
            tokenizer,
            prompt,
            args.device,
            args.tau,
            args.p,
        )
    atomic_to_json(serialize_channel_map(channel_map), output_dir / "identified_channels.json")

    targeted = select_top_layers(channel_map, args.top_layers)
    budget = sum(len(channels) for channels in targeted.values())
    random_channels = sample_random_channel_map(model, budget, args.seed)
    token_indices = tuple(args.token_indices)

    baseline = evaluate_dataframe(
        frame, model, tokenizer, args.device, args.max_new_tokens, "baseline"
    )
    targeted_config = channel_patch_configurations(targeted, token_indices)
    with patch_mixers(model, targeted_config):
        targeted_result = evaluate_dataframe(
            frame,
            model,
            tokenizer,
            args.device,
            args.max_new_tokens,
            "targeted intervention",
        )
    random_config = channel_patch_configurations(random_channels, token_indices)
    with patch_mixers(model, random_config):
        random_result = evaluate_dataframe(
            frame,
            model,
            tokenizer,
            args.device,
            args.max_new_tokens,
            "matched-random intervention",
        )

    results = {
        "baseline": baseline,
        "targeted_retention_channels": targeted_result,
        "matched_random_channels": random_result,
    }
    stats = pd.concat(
        [condition_stats(result, name, args.confidence) for name, result in results.items()],
        ignore_index=True,
    )
    summary = pd.DataFrame(
        [
            {
                "condition": name,
                "recall": float(result["is_correct"].mean()),
                "n_examples": len(result),
            }
            for name, result in results.items()
        ]
    )
    for name, result in results.items():
        atomic_to_csv(result, output_dir / f"{name}_per_example.csv")
    atomic_to_csv(stats, output_dir / "position_stats.csv")
    atomic_to_csv(summary, output_dir / "summary.csv")
    atomic_to_json(
        {
            "model": args.model,
            "dataset": str(args.dataset.expanduser().resolve()),
            "seed": args.seed,
            "tau": args.tau,
            "p": args.p,
            "top_layers": args.top_layers,
            "selected_layers": sorted(targeted),
            "channel_budget": budget,
            "token_indices_zero_based": list(token_indices),
            "limit": args.limit,
        },
        output_dir / "metadata.json",
    )
    save_plot(stats, output_dir / "recall_by_position.png")
    print(f"Saved intervention results to {output_dir}")


if __name__ == "__main__":
    main()
