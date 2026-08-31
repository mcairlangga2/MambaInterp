import numpy as np
import pandas as pd

from mambacode.evaluation import position_recall_stats, wilson_interval


def test_wilson_interval_is_bounded() -> None:
    lower, upper = wilson_interval(np.array([0, 5, 10]), np.array([10, 10, 10]))
    assert np.all(lower >= 0)
    assert np.all(upper <= 1)
    assert np.all(lower <= upper)


def test_position_recall_stats() -> None:
    frame = pd.DataFrame(
        {
            "input": ["a", "b", "c"],
            "target_position": [1, 1, 2],
            "completion": ["x", "x", "y"],
            "generated_text": ["x", "z", "y"],
            "is_correct": [True, False, True],
        }
    )
    stats = position_recall_stats(frame)
    assert stats["target_position"].tolist() == [1, 2]
    assert stats["n"].tolist() == [2, 1]
    assert stats["recall"].tolist() == [0.5, 1.0]
