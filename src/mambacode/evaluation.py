"""Shared evaluation, statistics, and output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


REQUIRED_DATA_COLUMNS = {"input", "target_position", "completion"}
REQUIRED_EVALUATION_COLUMNS = REQUIRED_DATA_COLUMNS | {
    "generated_text",
    "is_correct",
}


def load_recall_dataset(path: Path, limit: int | None = None) -> pd.DataFrame:
    """Load and validate one recall CSV."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    missing = REQUIRED_DATA_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Dataset is empty: {path}")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be a positive integer.")
        frame = frame.head(limit).copy()
    return frame


def evaluate_dataframe(
    frame: pd.DataFrame,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int = 2,
    description: str = "evaluation",
) -> pd.DataFrame:
    """Greedily generate completions and apply the original recall metric.

    A prediction is correct when the expected completion occurs as a
    case-insensitive substring of the newly generated text.
    """
    predictions: list[str] = []
    correctness: list[bool] = []

    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc=description):
        encoded = tokenizer(str(row.input), return_tensors="pt")
        encoded = {name: value.to(device) for name, value in encoded.items()}
        input_length = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0][input_length:],
            skip_special_tokens=True,
        )
        target = str(row.completion)
        predictions.append(generated)
        correctness.append(target.casefold() in generated.casefold())

    evaluated = frame.copy()
    evaluated["generated_text"] = predictions
    evaluated["is_correct"] = correctness
    return evaluated


def wilson_interval(
    successes: np.ndarray | pd.Series,
    totals: np.ndarray | pd.Series,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Wilson score intervals for binomial proportions."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    successes_array = np.asarray(successes, dtype=float)
    totals_array = np.asarray(totals, dtype=float)
    if np.any(totals_array <= 0):
        raise ValueError("totals must be positive.")

    p = successes_array / totals_array
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    denominator = 1.0 + (z**2) / totals_array
    center = (p + (z**2) / (2.0 * totals_array)) / denominator
    half_width = (z / denominator) * np.sqrt(
        (p * (1.0 - p) / totals_array)
        + (z**2) / (4.0 * totals_array**2)
    )
    return (
        np.clip(center - half_width, 0.0, 1.0),
        np.clip(center + half_width, 0.0, 1.0),
    )


def position_recall_stats(
    evaluated: pd.DataFrame,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Summarize recall and Wilson intervals by target position."""
    missing = REQUIRED_EVALUATION_COLUMNS.difference(evaluated.columns)
    if missing:
        raise ValueError(f"Evaluation is missing columns: {sorted(missing)}")
    stats = (
        evaluated.groupby("target_position", sort=True)["is_correct"]
        .agg(n="count", successes="sum", recall="mean")
        .reset_index()
    )
    lower, upper = wilson_interval(stats["successes"], stats["n"], confidence)
    stats["ci_lower"] = lower
    stats["ci_upper"] = upper
    stats["confidence"] = confidence
    return stats


def atomic_to_csv(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=index)
    temporary.replace(path)


def atomic_to_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_cached_evaluation(path: Path, expected_rows: int) -> pd.DataFrame | None:
    """Return a complete cached evaluation, or ``None`` if it is invalid."""
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    if len(frame) != expected_rows:
        return None
    if REQUIRED_EVALUATION_COLUMNS.difference(frame.columns):
        return None
    normalized = frame["is_correct"].astype(str).str.casefold()
    if not normalized.isin({"true", "false"}).all():
        return None
    frame["is_correct"] = normalized.eq("true")
    return frame
