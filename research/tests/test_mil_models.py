"""Checks using artificial inputs only; no study records are included."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dolphin_behavior_mil.mil_models import (  # noqa: E402
    build_cnn_mil_model,
    build_frozen_embedding_mil_model,
    trainable_parameter_count,
)


class MILModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.instances = (
            torch.randn(1, 128, 31),
            torch.randn(1, 128, 47),
            torch.randn(1, 128, 19),
        )
        self.bag_index = torch.tensor([0, 0, 1], dtype=torch.long)

    def test_all_five_pooling_heads_produce_encounter_logits(self) -> None:
        for model_id in (
            "cnn_mean",
            "cnn_max",
            "cnn_linear_softmax",
            "cnn_shared_gated_attention",
            "hf_lw_gam",
        ):
            with self.subTest(model_id=model_id):
                model = build_cnn_mil_model(
                    model_id,
                    base_channels=4,
                    embedding_dim=12,
                    attention_dim=7,
                    dropout=0.0,
                    encoder_microbatch_max_instances=2,
                )
                output = model(self.instances, self.bag_index, 2)
                self.assertEqual(output.logits.shape, (2, 4))
                self.assertEqual(output.instance_embeddings.shape, (3, 12))
                self.assertTrue(torch.all(torch.isfinite(output.logits)))
                output.logits.sum().backward()
                self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_label_wise_attention_sums_to_one_within_each_bag(self) -> None:
        model = build_cnn_mil_model(
            "hf_lw_gam",
            base_channels=4,
            embedding_dim=8,
            attention_dim=5,
            dropout=0.0,
        )
        output = model(self.instances, self.bag_index, 2)
        assert output.attention_weights is not None
        self.assertEqual(output.attention_weights.shape, (3, 4))
        for bag in range(2):
            sums = output.attention_weights[self.bag_index == bag].sum(dim=0)
            torch.testing.assert_close(sums, torch.ones(4))

    def test_variable_encoder_preserves_input_order(self) -> None:
        model = build_cnn_mil_model(
            "cnn_mean",
            base_channels=4,
            embedding_dim=8,
            attention_dim=5,
            dropout=0.0,
            encoder_microbatch_max_instances=2,
        )
        model.eval()
        with torch.no_grad():
            together = model.encoder.forward_variable(self.instances)
            separately = torch.cat(
                [model.encoder.forward_variable((value,)) for value in self.instances], dim=0
            )
        torch.testing.assert_close(together, separately, atol=1e-5, rtol=1e-5)

    def test_frozen_embedding_adapter_uses_same_labelwise_head(self) -> None:
        model = build_frozen_embedding_mil_model(
            "aves_frozen_mil", embedding_dim=6, attention_dim=4, dropout=0.0
        )
        values = (torch.randn(6), torch.randn(6), torch.randn(6))
        output = model(values, self.bag_index, 2)
        self.assertEqual(output.logits.shape, (2, 4))
        self.assertGreater(trainable_parameter_count(model), 0)
        self.assertFalse(any(True for _ in model.encoder.parameters()))

    def test_empty_bag_is_rejected(self) -> None:
        model = build_cnn_mil_model(
            "cnn_mean",
            base_channels=4,
            embedding_dim=8,
            attention_dim=4,
            dropout=0.0,
        )
        with self.assertRaises(ValueError):
            model(self.instances, torch.tensor([0, 0, 0]), 2)


if __name__ == "__main__":
    unittest.main()
