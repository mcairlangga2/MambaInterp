"""Falcon-Mamba channel interventions and SSM state initialization.

The modified slow recurrence follows Hugging Face Transformers 4.49.0's
``FalconMambaMixer.slow_forward`` (Apache-2.0) and exposes three controlled
changes: retention-score capture, recurrent-transition ablation, and nonzero
initial SSM state. See THIRD_PARTY_NOTICES.md.

Portions copyright 2024 Tri Dao, Albert Gu, Technological Innovation Institute,
and the Hugging Face team. Modifications copyright 2026 MambaCode contributors.
"""

from __future__ import annotations

import random
import types
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from typing import Iterator, Mapping, Sequence

import torch
from torch import nn

from .modeling import SUPPORTED_TRANSFORMERS_VERSION


@dataclass(frozen=True)
class CaptureConfig:
    tau: float = 0.7
    exclude_first: int = 1
    exclude_last: int = 2


@dataclass(frozen=True)
class StateInitConfig:
    mode: str = "uniform"
    value: float = 0.0


@dataclass(frozen=True)
class SlowForwardConfig:
    channels: tuple[int, ...] = ()
    token_indices: tuple[int, ...] = ()
    capture: CaptureConfig | None = None
    state_init: StateInitConfig | None = None


ChannelMap = dict[int, list[tuple[int, float]]]


def require_supported_transformers() -> None:
    installed = version("transformers")
    if installed != SUPPORTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "The causal experiments patch FalconMambaMixer internals and require "
            f"transformers=={SUPPORTED_TRANSFORMERS_VERSION}; found {installed}."
        )


def force_slow_path() -> None:
    """Disable optional fused kernels so the patched recurrence is exercised."""
    require_supported_transformers()
    from transformers.models.falcon_mamba import modeling_falcon_mamba

    modeling_falcon_mamba.is_fast_path_available = False


def get_mixers(model) -> list:
    try:
        mixers = [block.mixer for block in model.backbone.layers]
    except AttributeError as error:
        raise TypeError(
            "Causal experiments currently support Falcon-Mamba models with "
            "model.backbone.layers[*].mixer."
        ) from error
    if not mixers:
        raise ValueError("Model contains no Falcon-Mamba mixer layers.")
    return mixers


def _initialized_state(ssm_state: torch.Tensor, config: StateInitConfig) -> torch.Tensor:
    if config.mode == "uniform":
        return torch.rand_like(ssm_state)
    if config.mode == "normal":
        return torch.randn_like(ssm_state)
    if config.mode == "constant":
        return torch.full_like(ssm_state, config.value)
    raise ValueError(f"Unsupported state initialization mode: {config.mode}")


def make_modified_slow_forward(config: SlowForwardConfig):
    """Create a Transformers-4.49-compatible Falcon-Mamba slow recurrence."""
    from transformers.models.falcon_mamba.modeling_falcon_mamba import rms_forward

    channels = tuple(sorted(set(config.channels)))
    token_indices = tuple(sorted(set(config.token_indices)))

    def slow_forward(
        self,
        input_states,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
    ):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        projected_states = self.in_proj(input_states).transpose(1, 2)
        hidden_states, gate = projected_states.chunk(2, dim=1)
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        if cache_params is not None:
            ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            ssm_state = ssm_state.to(hidden_states.device)
            is_prefill = (
                cache_position is not None
                and cache_position.shape[0] == self.conv_kernel_size
            )
            if is_prefill:
                conv_state = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0),
                )
                cache_params.update_conv_state(
                    self.layer_idx,
                    conv_state,
                    cache_position,
                )
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])
            else:
                conv_state = cache_params.update_conv_state(
                    self.layer_idx,
                    hidden_states,
                    cache_position,
                )
                conv_state = conv_state.to(self.conv1d.weight.device)
                hidden_states = torch.sum(
                    conv_state * self.conv1d.weight[:, 0, :],
                    dim=-1,
                )
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)
        else:
            is_prefill = True
            ssm_state = torch.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size),
                device=hidden_states.device,
                dtype=dtype,
            )
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])

        if config.state_init is not None and is_prefill:
            ssm_state = _initialized_state(ssm_state, config.state_init)

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters,
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1,
        )
        B = rms_forward(B, variance_epsilon=self.rms_eps)
        C = rms_forward(C, variance_epsilon=self.rms_eps)
        time_step = rms_forward(time_step, variance_epsilon=self.rms_eps)

        discrete_time_step = self.dt_proj(time_step)
        discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2)
        A = -torch.exp(self.A_log.float())
        discrete_A = torch.exp(
            A[None, :, None, :] * discrete_time_step[:, :, :, None]
        )

        if config.capture is not None:
            start = config.capture.exclude_first
            stop = seq_len - config.capture.exclude_last
            if stop <= start:
                raise ValueError(
                    "The identification prompt is too short for the configured "
                    "capture exclusions."
                )
            products = torch.prod(discrete_A[:, :, start:stop, :], dim=2)
            retention_scores = (products > config.capture.tau).float().mean(dim=(0, 2))
            self._mambacode_retention_scores = retention_scores.detach().cpu()

        if channels and token_indices:
            valid_channels = [index for index in channels if 0 <= index < self.intermediate_size]
            if len(valid_channels) != len(channels):
                raise IndexError(
                    f"A channel index is outside [0, {self.intermediate_size - 1}]."
                )
            valid_tokens = [index for index in token_indices if 0 <= index < seq_len]
            if valid_tokens:
                for token_index in valid_tokens:
                    discrete_A[:, valid_channels, token_index, :] = 0

        discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

        scan_outputs = []
        for index in range(seq_len):
            ssm_state = (
                discrete_A[:, :, index, :] * ssm_state
                + deltaB_u[:, :, index, :]
            )
            scan_output = torch.matmul(
                ssm_state.to(dtype),
                C[:, index, :].unsqueeze(-1),
            )
            scan_outputs.append(scan_output[:, :, 0])

        scan_output = torch.stack(scan_outputs, dim=-1)
        scan_output = scan_output + (hidden_states * self.D[None, :, None])
        scan_output = scan_output * self.act(gate)
        if cache_params is not None:
            cache_params.update_ssm_state(self.layer_idx, ssm_state)
        return self.out_proj(scan_output.transpose(1, 2))

    return slow_forward


@contextmanager
def patch_mixers(
    model,
    configurations: Mapping[int, SlowForwardConfig],
) -> Iterator[None]:
    """Temporarily patch selected mixer layers and always restore them."""
    mixers = get_mixers(model)
    originals: dict[int, object] = {}
    try:
        for layer_index, config in configurations.items():
            if not 0 <= layer_index < len(mixers):
                raise IndexError(
                    f"Layer {layer_index} is outside [0, {len(mixers) - 1}]."
                )
            mixer = mixers[layer_index]
            originals[layer_index] = mixer.slow_forward
            mixer.slow_forward = types.MethodType(
                make_modified_slow_forward(config),
                mixer,
            )
        yield
    finally:
        for layer_index, original in originals.items():
            mixers[layer_index].slow_forward = original


def identify_retention_channels(
    model,
    tokenizer,
    prompt: str,
    device: str,
    tau: float = 0.7,
    p: float = 0.7,
) -> ChannelMap:
    """Identify channels using the recurrence-retention criterion.

    For each layer and channel, multiply ``A_t`` over context tokens (excluding
    the first context token and final two query tokens). A channel is retained
    when more than ``p`` of its state dimensions have a product above ``tau``.
    """
    if not 0.0 <= tau <= 1.0 or not 0.0 <= p <= 1.0:
        raise ValueError("tau and p must be between 0 and 1.")
    mixers = get_mixers(model)
    capture = CaptureConfig(tau=tau)
    configs = {
        layer_index: SlowForwardConfig(capture=capture)
        for layer_index in range(len(mixers))
    }
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {name: value.to(device) for name, value in encoded.items()}
    with patch_mixers(model, configs), torch.inference_mode():
        model(**encoded, use_cache=False)

    channel_map: ChannelMap = {}
    for layer_index, mixer in enumerate(mixers):
        scores = getattr(mixer, "_mambacode_retention_scores", None)
        if scores is None:
            raise RuntimeError(f"No retention scores captured for layer {layer_index}.")
        selected = torch.nonzero(scores > p, as_tuple=False).flatten().tolist()
        channel_map[layer_index] = [
            (int(channel), float(scores[channel])) for channel in selected
        ]
        delattr(mixer, "_mambacode_retention_scores")
    return channel_map


def select_top_layers(channel_map: ChannelMap, count: int) -> dict[int, list[int]]:
    if count < 1:
        raise ValueError("top layer count must be positive.")
    ordered = sorted(channel_map, key=lambda layer: (-len(channel_map[layer]), layer))
    selected: dict[int, list[int]] = {}
    for layer in ordered[:count]:
        channels = [channel for channel, _ in channel_map[layer]]
        if channels:
            selected[layer] = channels
    if not selected:
        raise ValueError("No channels met the retention criterion.")
    return selected


def sample_random_channel_map(
    model,
    budget: int,
    seed: int,
) -> dict[int, list[int]]:
    mixers = get_mixers(model)
    intermediate_size = int(mixers[0].intermediate_size)
    population = len(mixers) * intermediate_size
    if not 0 < budget <= population:
        raise ValueError(f"budget must be between 1 and {population}.")
    flat_indices = random.Random(seed).sample(range(population), budget)
    sampled: dict[int, list[int]] = {}
    for flat_index in flat_indices:
        layer, channel = divmod(flat_index, intermediate_size)
        sampled.setdefault(layer, []).append(channel)
    return {layer: sorted(channels) for layer, channels in sorted(sampled.items())}


def channel_patch_configurations(
    channels_by_layer: Mapping[int, Sequence[int]],
    token_indices: Sequence[int],
) -> dict[int, SlowForwardConfig]:
    return {
        int(layer): SlowForwardConfig(
            channels=tuple(int(channel) for channel in channels),
            token_indices=tuple(int(index) for index in token_indices),
        )
        for layer, channels in channels_by_layer.items()
    }


def serialize_channel_map(channel_map: ChannelMap) -> dict[str, list[dict[str, float | int]]]:
    return {
        str(layer): [
            {"channel": channel, "retention_score": score}
            for channel, score in entries
        ]
        for layer, entries in sorted(channel_map.items())
    }


def deserialize_channel_map(
    payload: Mapping[str, Sequence[Mapping[str, float | int]]],
) -> ChannelMap:
    return {
        int(layer): [
            (int(entry["channel"]), float(entry["retention_score"]))
            for entry in entries
        ]
        for layer, entries in payload.items()
    }
