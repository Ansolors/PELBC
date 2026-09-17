"""Small high-frequency CNN and comparable multiple-instance pooling heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn


POOLING_IDS = (
    "mean",
    "max",
    "linear_softmax",
    "shared_gated_attention",
    "label_wise_gated_attention",
)


class ChannelLayerNorm2d(nn.Module):
    """Normalize channels at each time-frequency location without reading padding."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.normalization(inputs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            ChannelLayerNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )


class SmallWhistleCNN(nn.Module):
    """Encode uncropped variable-duration [1, 128, time] whistle spectrograms."""

    def __init__(
        self,
        *,
        base_channels: int,
        embedding_dim: int,
        dropout: float,
        encoder_microbatch_max_instances: int = 32,
    ) -> None:
        super().__init__()
        if base_channels < 1 or embedding_dim < 1:
            raise ValueError("channel and embedding dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if encoder_microbatch_max_instances < 1:
            raise ValueError("encoder microbatch size must be positive")
        self.base_channels = int(base_channels)
        self.embedding_dim = int(embedding_dim)
        self.encoder_microbatch_max_instances = int(encoder_microbatch_max_instances)
        channels = (base_channels, 2 * base_channels, 4 * base_channels)
        self.blocks = nn.ModuleList(
            (
            ConvBlock(1, channels[0]),
            ConvBlock(channels[0], channels[1]),
            ConvBlock(channels[1], channels[2]),
            )
        )
        self.projection = nn.Sequential(
            nn.Linear(channels[-1], embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    @staticmethod
    def output_time_lengths(lengths: torch.Tensor) -> torch.Tensor:
        result = lengths.to(torch.long)
        for _ in range(3):
            result = torch.div(result + 1, 2, rounding_mode="floor")
        return result

    def forward_padded(
        self,
        inputs: torch.Tensor,
        time_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 1 or inputs.shape[2] != 128:
            raise ValueError("CNN input must have shape [instance, 1, 128, time]")
        if time_lengths.shape != (len(inputs),):
            raise ValueError("one time length is required per instance")
        if torch.any(time_lengths < 1) or torch.any(time_lengths > inputs.shape[-1]):
            raise ValueError("time lengths must lie within padded input bounds")
        encoded = inputs
        valid_lengths = time_lengths.to(torch.long)
        for block in self.blocks:
            encoded = block(encoded)
            valid_lengths = torch.div(valid_lengths + 1, 2, rounding_mode="floor")
            block_time_index = torch.arange(encoded.shape[-1], device=encoded.device)[None, :]
            block_mask = block_time_index < valid_lengths[:, None]
            encoded = encoded * block_mask[:, None, None, :].to(encoded.dtype)
        time_index = torch.arange(encoded.shape[-1], device=encoded.device)[None, :]
        mask = time_index < valid_lengths[:, None]
        weighted = encoded * mask[:, None, None, :].to(encoded.dtype)
        denominator = valid_lengths.to(encoded.dtype) * encoded.shape[2]
        pooled = weighted.sum(dim=(2, 3)) / denominator[:, None].clamp_min(1.0)
        return self.projection(pooled)

    def forward_variable(self, instances: Sequence[torch.Tensor]) -> torch.Tensor:
        if not instances:
            raise ValueError("at least one whistle instance is required")
        for value in instances:
            if value.ndim != 3 or value.shape[:2] != (1, 128) or value.shape[-1] < 1:
                raise ValueError("each instance must have shape [1, 128, time]")
        devices = {str(value.device) for value in instances}
        if len(devices) != 1:
            raise ValueError("all instances must reside on the same device")
        ordered_indices = sorted(range(len(instances)), key=lambda index: instances[index].shape[-1])
        output: list[torch.Tensor | None] = [None] * len(instances)
        maximum = self.encoder_microbatch_max_instances
        for start in range(0, len(ordered_indices), maximum):
            indices = ordered_indices[start : start + maximum]
            values = [instances[index] for index in indices]
            lengths = torch.tensor(
                [value.shape[-1] for value in values],
                dtype=torch.long,
                device=values[0].device,
            )
            padded = values[0].new_zeros((len(values), 1, 128, int(lengths.max().item())))
            for row_index, value in enumerate(values):
                padded[row_index, :, :, : value.shape[-1]] = value
            encoded = self.forward_padded(padded, lengths)
            for row_index, original_index in enumerate(indices):
                output[original_index] = encoded[row_index]
        if any(value is None for value in output):  # pragma: no cover - defensive
            raise RuntimeError("variable encoder failed to restore instance ordering")
        return torch.stack([value for value in output if value is not None])


class IdentityEmbeddingEncoder(nn.Module):
    """Adapter for precomputed frozen PANNs/AVES vectors."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = int(embedding_dim)

    def forward_variable(self, instances: Sequence[torch.Tensor]) -> torch.Tensor:
        values = []
        for value in instances:
            flattened = value.reshape(-1)
            if flattened.shape != (self.embedding_dim,):
                raise ValueError(
                    f"expected frozen embedding dimension {self.embedding_dim}, got {flattened.shape}"
                )
            values.append(flattened)
        if not values:
            raise ValueError("at least one frozen embedding is required")
        return torch.stack(values)


class GatedAttentionScores(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int, output_count: int) -> None:
        super().__init__()
        if min(embedding_dim, attention_dim, output_count) < 1:
            raise ValueError("attention dimensions must be positive")
        self.value = nn.Linear(embedding_dim, attention_dim)
        self.gate = nn.Linear(embedding_dim, attention_dim)
        self.output = nn.Linear(attention_dim, output_count)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.value(embeddings)) * torch.sigmoid(self.gate(embeddings))
        return self.output(hidden)


@dataclass(frozen=True)
class MILOutput:
    logits: torch.Tensor
    instance_embeddings: torch.Tensor
    attention_weights: torch.Tensor | None
    instance_logits: torch.Tensor | None


class MILClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        *,
        embedding_dim: int,
        label_count: int = 4,
        pooling: str,
        attention_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if pooling not in POOLING_IDS:
            raise ValueError(f"unknown MIL pooling {pooling}")
        if min(embedding_dim, label_count, attention_dim) < 1:
            raise ValueError("MIL dimensions must be positive")
        self.encoder = encoder
        self.embedding_dim = int(embedding_dim)
        self.label_count = int(label_count)
        self.pooling = str(pooling)
        self.dropout = nn.Dropout(float(dropout))
        if pooling in {"mean", "max"}:
            self.classifier = nn.Linear(embedding_dim, label_count)
            self.attention = None
        elif pooling == "linear_softmax":
            self.classifier = nn.Linear(embedding_dim, label_count)
            self.attention = None
        elif pooling == "shared_gated_attention":
            self.attention = GatedAttentionScores(embedding_dim, attention_dim, 1)
            self.classifier = nn.Linear(embedding_dim, label_count)
        else:
            self.attention = GatedAttentionScores(embedding_dim, attention_dim, label_count)
            self.classifier_weight = nn.Parameter(torch.empty(label_count, embedding_dim))
            self.classifier_bias = nn.Parameter(torch.zeros(label_count))
            nn.init.xavier_uniform_(self.classifier_weight)

    @staticmethod
    def _validate_bags(
        bag_index: torch.Tensor,
        instance_count: int,
        bag_count: int,
    ) -> None:
        if bag_index.ndim != 1 or len(bag_index) != instance_count:
            raise ValueError("bag_index must contain one value per instance")
        if bag_count < 1 or torch.any(bag_index < 0) or torch.any(bag_index >= bag_count):
            raise ValueError("bag indices lie outside the declared bag count")
        counts = torch.bincount(bag_index, minlength=bag_count)
        if torch.any(counts == 0):
            raise ValueError("every bag must contain at least one instance")

    def forward(
        self,
        instances: Sequence[torch.Tensor],
        bag_index: torch.Tensor,
        bag_count: int,
    ) -> MILOutput:
        embeddings = self.encoder.forward_variable(instances)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise RuntimeError("encoder returned an incompatible embedding matrix")
        self._validate_bags(bag_index, len(embeddings), bag_count)
        dropped = self.dropout(embeddings)
        attention_weights: torch.Tensor | None = None
        instance_logits: torch.Tensor | None = None

        if self.pooling in {"mean", "max"}:
            pooled = []
            for bag in range(bag_count):
                values = dropped[bag_index == bag]
                pooled.append(values.mean(dim=0) if self.pooling == "mean" else values.max(dim=0).values)
            logits = self.classifier(torch.stack(pooled))
        elif self.pooling == "linear_softmax":
            instance_logits = self.classifier(dropped)
            probabilities = torch.sigmoid(instance_logits)
            pooled_probabilities = []
            weights = torch.zeros_like(probabilities)
            for bag in range(bag_count):
                mask = bag_index == bag
                values = probabilities[mask]
                denominator = values.sum(dim=0).clamp_min(1.0e-7)
                pooled_probabilities.append((values.square().sum(dim=0) / denominator).clamp(1.0e-6, 1.0 - 1.0e-6))
                weights[mask] = values / denominator
            pooled_probability = torch.stack(pooled_probabilities)
            logits = torch.logit(pooled_probability)
            attention_weights = weights
        elif self.pooling == "shared_gated_attention":
            assert self.attention is not None
            scores = self.attention(dropped).squeeze(1)
            weights = torch.zeros_like(scores)
            pooled = []
            for bag in range(bag_count):
                mask = bag_index == bag
                local_weights = torch.softmax(scores[mask], dim=0)
                weights[mask] = local_weights
                pooled.append(torch.sum(local_weights[:, None] * dropped[mask], dim=0))
            logits = self.classifier(torch.stack(pooled))
            attention_weights = weights[:, None]
        else:
            assert self.attention is not None
            scores = self.attention(dropped)
            weights = torch.zeros_like(scores)
            pooled_bags = []
            for bag in range(bag_count):
                mask = bag_index == bag
                local_weights = torch.softmax(scores[mask], dim=0)
                weights[mask] = local_weights
                pooled_bags.append(
                    torch.einsum("nl,nd->ld", local_weights, dropped[mask])
                )
            pooled = torch.stack(pooled_bags)
            logits = torch.einsum("bld,ld->bl", pooled, self.classifier_weight)
            logits = logits + self.classifier_bias[None, :]
            attention_weights = weights

        return MILOutput(
            logits=logits,
            instance_embeddings=embeddings,
            attention_weights=attention_weights,
            instance_logits=instance_logits,
        )


MODEL_TO_POOLING = {
    "cnn_mean": "mean",
    "cnn_max": "max",
    "cnn_linear_softmax": "linear_softmax",
    "cnn_shared_gated_attention": "shared_gated_attention",
    "hf_lw_gam": "label_wise_gated_attention",
    "panns_frozen_mil": "label_wise_gated_attention",
    "aves_frozen_mil": "label_wise_gated_attention",
}


def build_cnn_mil_model(
    model_id: str,
    *,
    base_channels: int,
    embedding_dim: int,
    attention_dim: int,
    dropout: float,
    encoder_microbatch_max_instances: int = 32,
) -> MILClassifier:
    if model_id not in {
        "cnn_mean",
        "cnn_max",
        "cnn_linear_softmax",
        "cnn_shared_gated_attention",
        "hf_lw_gam",
    }:
        raise ValueError(f"{model_id} is not a CNN MIL model")
    encoder = SmallWhistleCNN(
        base_channels=base_channels,
        embedding_dim=embedding_dim,
        dropout=dropout,
        encoder_microbatch_max_instances=encoder_microbatch_max_instances,
    )
    return MILClassifier(
        encoder,
        embedding_dim=embedding_dim,
        pooling=MODEL_TO_POOLING[model_id],
        attention_dim=attention_dim,
        dropout=dropout,
    )


def build_frozen_embedding_mil_model(
    model_id: str,
    *,
    embedding_dim: int,
    attention_dim: int,
    dropout: float,
) -> MILClassifier:
    if model_id not in {"panns_frozen_mil", "aves_frozen_mil"}:
        raise ValueError(f"{model_id} is not a frozen embedding model")
    return MILClassifier(
        IdentityEmbeddingEncoder(embedding_dim),
        embedding_dim=embedding_dim,
        pooling="label_wise_gated_attention",
        attention_dim=attention_dim,
        dropout=dropout,
    )


def trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
