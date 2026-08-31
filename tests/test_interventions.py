from types import SimpleNamespace

from mambacode.interventions import (
    deserialize_channel_map,
    sample_random_channel_map,
    select_top_layers,
    serialize_channel_map,
)


def toy_model(number_of_layers: int = 3, intermediate_size: int = 5):
    layers = [
        SimpleNamespace(mixer=SimpleNamespace(intermediate_size=intermediate_size))
        for _ in range(number_of_layers)
    ]
    return SimpleNamespace(backbone=SimpleNamespace(layers=layers))


def test_select_top_layers_and_serialization() -> None:
    channel_map = {0: [(1, 0.8)], 1: [(0, 0.9), (2, 0.85)], 2: []}
    assert select_top_layers(channel_map, 1) == {1: [0, 2]}
    assert deserialize_channel_map(serialize_channel_map(channel_map)) == channel_map


def test_random_channel_budget_is_exact_and_unique() -> None:
    sampled = sample_random_channel_map(toy_model(), budget=7, seed=42)
    pairs = [(layer, channel) for layer, channels in sampled.items() for channel in channels]
    assert len(pairs) == 7
    assert len(set(pairs)) == 7
