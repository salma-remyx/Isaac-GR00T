# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test task-conditioned, temporally adaptive visual token pruning (FastVLA-style).

The integration tests go through Gr00tN1d7ActionHead's existing
process_backbone_output / _encode_features / get_action paths with the opt-in
``visual_token_pruner_cfg`` config key, proving the pruned visual stream flows
through the unmodified vl_self_attention and DiT consumers. The unit tests
exercise the pruner's scoring, adaptive budget, and compaction directly.
"""

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from gr00t.model.modules.visual_token_pruner import AdaptiveVisualTokenPruner
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


# 4 task tokens + 12 visual tokens; the first two steps of an episode are
# FastVLA's warm-up (rate = rho - tau, the maximum-keep end of the band), so a
# fresh pruner keeps at max_keep_ratio 0.75: round(0.75 * 12) = 9 visual
# tokens kept -> 4 + 9 = 13 total.
NUM_TASK_TOKENS = 4
SEQ_LEN = 16
EXPECTED_PRUNED_LEN = 13

_PRUNER_CFG = {
    "min_keep_ratio": 0.25,
    "max_keep_ratio": 0.75,
    "min_visual_tokens": 2,
    "window_size": 4,
}


def _small_config(**overrides) -> Gr00tN1d7Config:
    defaults = dict(
        backbone_embedding_dim=64,
        hidden_size=64,
        input_embedding_dim=64,
        max_state_dim=7,
        max_action_dim=7,
        action_horizon=4,
        state_history_length=1,
        num_inference_timesteps=2,
        max_num_embodiments=4,
        add_pos_embed=True,
        use_vlln=True,
        max_seq_len=32,
        use_alternate_vl_dit=False,
        attend_text_every_n_blocks=2,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        state_dropout_prob=0.0,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        attn_dropout=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 32,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 64,
            "interleave_self_attention": True,
        },
    )
    defaults.update(overrides)
    return Gr00tN1d7Config(**defaults)


def _pruned_config(**overrides) -> Gr00tN1d7Config:
    pruner_cfg = overrides.pop("visual_token_pruner_cfg", dict(_PRUNER_CFG))
    return _small_config(visual_token_pruner_cfg=pruner_cfg, **overrides)


def _make_backbone_output(config, batch_size=2, seq_len=SEQ_LEN):
    torch.manual_seed(0)
    image_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    image_mask[:, NUM_TASK_TOKENS:] = True
    return BatchFeature(
        data={
            "backbone_features": torch.randn(batch_size, seq_len, config.backbone_embedding_dim),
            "backbone_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "image_mask": image_mask,
        }
    )


def _make_action_input(config, batch_size=2):
    return BatchFeature(
        data={
            "state": torch.randn(batch_size, config.state_history_length, config.max_state_dim),
            "action": torch.randn(batch_size, config.action_horizon, config.max_action_dim),
            "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
            "action_mask": torch.ones(batch_size, config.action_horizon, config.max_action_dim),
        }
    )


class TestPruningWiredIntoActionHead:
    """Integration: the opt-in config key prunes the head's visual stream."""

    def test_encode_features_prunes_visual_stream(self):
        config = _pruned_config()
        head = Gr00tN1d7ActionHead(config)
        head.eval()
        backbone_output = _make_backbone_output(config)

        result = head._encode_features(backbone_output, _make_action_input(config))

        stats = head.visual_token_pruner.last_stats
        assert stats is not None
        assert stats.in_tokens == SEQ_LEN
        assert stats.keep_ratio == pytest.approx(0.75)  # warm-up step: max keep
        assert stats.out_tokens == EXPECTED_PRUNED_LEN

        # The returned features and the in-place rewritten masks agree on the
        # pruned length, all task tokens survive, and the visual token count
        # matches the adaptive budget.
        assert result["backbone_features"].shape == (
            2,
            EXPECTED_PRUNED_LEN,
            config.backbone_embedding_dim,
        )
        assert backbone_output["image_mask"].shape[1] == EXPECTED_PRUNED_LEN
        assert backbone_output["backbone_attention_mask"].shape[1] == EXPECTED_PRUNED_LEN
        assert (
            backbone_output["image_mask"].sum(dim=1).eq(EXPECTED_PRUNED_LEN - NUM_TASK_TOKENS).all()
        )
        assert (~backbone_output["image_mask"]).sum(dim=1).eq(NUM_TASK_TOKENS).all()
        assert backbone_output["backbone_attention_mask"].all()

    def test_get_action_runs_with_pruning_through_alternate_dit(self):
        config = _pruned_config(use_alternate_vl_dit=True)
        head = Gr00tN1d7ActionHead(config)
        head.eval()
        action_input = _make_action_input(config)
        del action_input["action"]

        out = head.get_action(_make_backbone_output(config), action_input)

        # The AlternateVLDiT image/text mask path accepted the shorter stream.
        assert head.visual_token_pruner.last_stats.out_tokens == EXPECTED_PRUNED_LEN
        assert out["action_pred"].shape == (2, config.action_horizon, config.max_action_dim)
        assert torch.isfinite(out["action_pred"]).all()

    def test_disabled_by_default(self):
        head = Gr00tN1d7ActionHead(_small_config())
        head.eval()
        backbone_output = _make_backbone_output(head.config)

        head._encode_features(backbone_output, _make_action_input(head.config))

        assert head.visual_token_pruner is None
        assert backbone_output["backbone_features"].shape[1] == SEQ_LEN

    def test_training_forward_not_pruned_by_default(self):
        config = _pruned_config()
        head = Gr00tN1d7ActionHead(config)
        head.train()
        backbone_output = _make_backbone_output(config)

        out = head.forward(backbone_output, _make_action_input(config))

        assert torch.isfinite(out["loss"])
        assert head.visual_token_pruner.last_stats is None  # training-free: eval only
        assert backbone_output["backbone_features"].shape[1] == SEQ_LEN

    def test_training_forward_pruned_when_opted_in(self):
        config = _pruned_config(
            visual_token_pruner_cfg={**_PRUNER_CFG, "apply_during_training": True}
        )
        head = Gr00tN1d7ActionHead(config)
        head.train()
        backbone_output = _make_backbone_output(config)

        out = head.forward(backbone_output, _make_action_input(config))

        assert torch.isfinite(out["loss"])
        assert backbone_output["backbone_features"].shape[1] == EXPECTED_PRUNED_LEN


def _make_stream(features: torch.Tensor, image_flags: list[bool]) -> BatchFeature:
    batch_size, seq_len = features.shape[0], features.shape[1]
    image_mask = torch.tensor(image_flags, dtype=torch.bool).unsqueeze(0).expand(batch_size, -1)
    return BatchFeature(
        data={
            "backbone_features": features,
            "backbone_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "image_mask": image_mask,
        }
    )


def _distribution_stream(kind: str, num_visual: int = 8, dim: int = 4) -> BatchFeature:
    """One task token followed by visual tokens with peaked/uniform importance.

    The task token points along the first basis vector; in the peaked case a
    single visual token is strongly aligned with it while the rest are
    orthogonal, in the uniform case all visual tokens are identical.
    """
    task = [1.0] + [0.0] * (dim - 1)
    if kind == "peaked":
        visual = [[4.0] + [0.0] * (dim - 1)] + [[0.0, 0.5] + [0.0] * (dim - 2)] * (num_visual - 1)
    else:
        visual = [[0.3] + [0.0] * (dim - 1)] * num_visual
    features = torch.tensor([task] + visual, dtype=torch.float32).unsqueeze(0)
    return _make_stream(features, [False] + [True] * num_visual)


class TestAdaptiveBudget:
    """FastVLA's inverse-Simpson uniformity + sliding-window calibration."""

    def test_flat_distribution_keeps_more_than_peaked(self):
        pruner = AdaptiveVisualTokenPruner(
            min_keep_ratio=0.2, max_keep_ratio=0.8, min_visual_tokens=1, window_size=4
        )

        # The first two steps are warm-up (max keep) whatever the distribution.
        pruner(_distribution_stream("peaked"))
        warmup_stats = pruner.last_stats
        assert warmup_stats.keep_ratio == pytest.approx(0.8)

        pruner(_distribution_stream("uniform"))
        assert pruner.last_stats.keep_ratio == pytest.approx(0.8)  # still warm-up

        # From step 2 the window calibrates: uniform importance is scored more
        # uniform than peaked importance, and the budget follows: max for the
        # flattest step in the window, min for the most peaked one.
        pruner(_distribution_stream("peaked"))
        peaked_stats = pruner.last_stats
        pruner(_distribution_stream("uniform"))
        uniform_stats = pruner.last_stats

        assert uniform_stats.uniformity > peaked_stats.uniformity
        assert peaked_stats.keep_ratio == pytest.approx(0.2)
        assert uniform_stats.keep_ratio == pytest.approx(0.8)
        assert peaked_stats.keep_ratio < warmup_stats.keep_ratio

    def test_warmup_uses_max_keep_for_first_two_steps(self):
        # FastVLA's warm-up: while the calibration window is unpopulated, the
        # first two steps of every episode run at rate = rho - tau, the
        # maximum-keep end of the band (reference test: kept[0] == kept[1] == hi).
        pruner = AdaptiveVisualTokenPruner(
            min_keep_ratio=0.2, max_keep_ratio=0.8, min_visual_tokens=1, window_size=4
        )
        torch.manual_seed(0)
        for _ in range(2):
            pruner(_make_stream(torch.randn(1, 9, 4), [False] + [True] * 8))
            assert pruner.last_stats.keep_ratio == pytest.approx(0.8)

        # Step 2 calibrates against the window of the two warm-up steps.
        pruner(_make_stream(torch.randn(1, 9, 4), [False] + [True] * 8))
        assert 0.2 <= pruner.last_stats.keep_ratio <= 0.8

    def test_uniformity_is_computed_over_keep_set(self):
        # FastVLA's uniformity is the normalized inverse-Simpson index of the
        # top-K importance scores (K = smallest budget the band can request),
        # not of all N scores: importance uniform over the eventual keep-set
        # with a near-zero tail must yield U ~ 1, not U ~ min_keep_ratio.
        pruner = AdaptiveVisualTokenPruner(min_keep_ratio=0.25, max_keep_ratio=0.75)
        scores = torch.tensor([[9.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])

        _, uniformity, _ = pruner._budget(torch.tensor([8]), scores)

        # K = ceil(8 * 0.25 - 1e-9) = 2; q = [0.6, 0.4] over the keep set.
        assert uniformity == pytest.approx((1.0 / (0.6**2 + 0.4**2)) / 2)
        all_token = (1.0 / ((9 / 21) ** 2 + (6 / 21) ** 2 + 6 * (1 / 21) ** 2)) / 8
        assert uniformity > all_token + 0.4  # all-token metric would be ~0.45

    def test_keep_count_respects_bounds(self):
        pruner = AdaptiveVisualTokenPruner(
            min_keep_ratio=0.25, max_keep_ratio=0.75, min_visual_tokens=3, window_size=4
        )
        torch.manual_seed(1)
        for _ in range(6):
            stream = _make_stream(torch.randn(2, 16, 8), [False] * 4 + [True] * 12)
            num_visual = int(stream["image_mask"][0].sum())
            pruner(stream)
            stats = pruner.last_stats
            kept_visual = int(stream["image_mask"].sum(dim=1).min())
            # Budget stays inside the configured ratio band, and the kept count
            # inside [min_visual_tokens floor, available visual tokens].
            assert 0.25 <= stats.keep_ratio <= 0.75
            assert min(3, num_visual) <= kept_visual <= num_visual

    def test_reset_clears_calibration(self):
        pruner = AdaptiveVisualTokenPruner(
            min_keep_ratio=0.2, max_keep_ratio=0.8, min_visual_tokens=1, window_size=4
        )
        pruner(_distribution_stream("uniform"))
        pruner(_distribution_stream("peaked"))
        assert pruner.last_stats is not None

        pruner.reset()
        assert pruner.last_stats is None
        pruner(_distribution_stream("uniform"))
        # Window is empty and the step counter restarts, so the warm-up
        # (max-keep) rule applies again — reset makes episodes deterministic.
        assert pruner.last_stats.keep_ratio == pytest.approx(0.8)


class TestSelectionAndCompaction:
    """Top-k selection, task-token preservation, and mask consistency."""

    def test_task_tokens_kept_first_in_order(self):
        torch.manual_seed(0)
        features = torch.randn(1, 6, 8)
        stream = _make_stream(features, [False, False, True, True, True, True])

        pruner = AdaptiveVisualTokenPruner(
            min_keep_ratio=0.25, max_keep_ratio=0.75, min_visual_tokens=1, window_size=4
        )
        pruner(stream)

        # Warm-up step -> max ratio 0.75 -> round(0.75 * 4) = 3 visual kept.
        assert stream["backbone_features"].shape[1] == 5
        assert torch.equal(stream["backbone_features"][0, :2], features[0, :2])
        assert stream["image_mask"][0].tolist() == [False, False, True, True, True]
        assert stream["backbone_attention_mask"].all()

    def test_keep_count_uses_round_not_ceil(self):
        # FastVLA keeps round((1 - rate) * N) tokens; for N = 8 and warm-up
        # ratio 0.3 that is round(2.4) = 2, where a ceil would keep 3.
        torch.manual_seed(0)
        stream = _make_stream(torch.randn(1, 9, 4), [False] + [True] * 8)

        pruner = AdaptiveVisualTokenPruner(
            min_keep_ratio=0.2, max_keep_ratio=0.3, min_visual_tokens=1, window_size=4
        )
        pruner(stream)

        assert stream["backbone_features"].shape[1] == 3  # 1 task + 2 visual

    def test_highest_importance_visual_token_survives(self):
        stream = _distribution_stream("peaked")
        aligned = torch.tensor([4.0, 0.0, 0.0, 0.0])

        pruner = AdaptiveVisualTokenPruner(
            min_keep_ratio=0.2, max_keep_ratio=0.3, min_visual_tokens=1, window_size=4
        )
        pruner(stream)

        kept = stream["backbone_features"][0]
        # The single task-aligned visual token is the top-scored one; it must be
        # among the survivors even at the most aggressive budget.
        assert torch.isclose(kept, aligned).all(dim=1).any()
        assert stream["image_mask"].sum() >= 1

    def test_noop_without_image_mask(self):
        torch.manual_seed(0)
        features = torch.randn(2, 6, 8)
        stream = BatchFeature(
            data={
                "backbone_features": features.clone(),
                "backbone_attention_mask": torch.ones(2, 6, dtype=torch.bool),
            }
        )

        pruner = AdaptiveVisualTokenPruner()
        pruner(stream)

        assert torch.equal(stream["backbone_features"], features)
        assert pruner.last_stats is None

    def test_noop_when_sample_has_no_task_tokens(self):
        torch.manual_seed(0)
        features = torch.randn(2, 6, 8)
        stream = _make_stream(features, [True] * 6)  # every token visual

        pruner = AdaptiveVisualTokenPruner()
        pruner(stream)

        assert torch.equal(stream["backbone_features"], features)
        assert pruner.last_stats is None


class TestConfigValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_keep_ratio": 0.8, "max_keep_ratio": 0.2},
            {"min_keep_ratio": 0.0, "max_keep_ratio": 0.5},
            {"min_visual_tokens": 0},
        ],
    )
    def test_invalid_config_raises(self, kwargs):
        defaults = {"min_keep_ratio": 0.25, "max_keep_ratio": 0.75}
        defaults.update(kwargs)
        with pytest.raises(ValueError):
            AdaptiveVisualTokenPruner(**defaults)

    def test_window_size_one_is_allowed(self):
        # The reference allows window >= 1 (a 1-step window degenerates to the
        # midpoint budget after warm-up); only < 1 is invalid.
        AdaptiveVisualTokenPruner(window_size=1)
        with pytest.raises(ValueError):
            AdaptiveVisualTokenPruner(window_size=0)


class TestImportanceScoring:
    """FastVLA's parameter-free task-to-vision attention over RMSNorm'd features."""

    def test_importance_uses_direction_not_raw_norm(self):
        # Two near-aligned visual tokens: a tiny-norm but better-aligned one
        # and a large-norm but less-aligned one. RMSNorm makes the score track
        # direction, so the aligned token wins despite its 30x smaller norm;
        # raw dot products would let the large norm dominate instead.
        features = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.1, 0.1, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0]]]
        )
        stream = _make_stream(features, [False, True, True])

        pruner = AdaptiveVisualTokenPruner(
            min_keep_ratio=0.2, max_keep_ratio=0.3, min_visual_tokens=1, window_size=4
        )
        pruner(stream)

        # Warm-up ratio 0.3 -> round(0.3 * 2) = 1 visual token kept: the
        # better-aligned one, not the larger-normed one.
        assert stream["backbone_features"].shape[1] == 2
        assert torch.equal(stream["backbone_features"][0, 1], features[0, 1])

    def test_zero_task_rows_do_not_change_importance(self):
        # Zero-vector task rows are placeholders and must be skipped, exactly
        # as in the reference's importance().
        torch.manual_seed(0)
        task, visual = torch.randn(2, 4), torch.randn(6, 4)
        base = torch.cat([task, visual]).unsqueeze(0)
        base_task = torch.zeros(1, 8, dtype=torch.bool)
        base_task[0, :2] = True
        base_visual = torch.zeros(1, 8, dtype=torch.bool)
        base_visual[0, 2:] = True

        padded = torch.cat([task, torch.zeros(2, 4), visual]).unsqueeze(0)
        padded_task = torch.zeros(1, 10, dtype=torch.bool)
        padded_task[0, :4] = True  # zero rows sit inside the task span
        padded_visual = torch.zeros(1, 10, dtype=torch.bool)
        padded_visual[0, 4:] = True

        base_scores = AdaptiveVisualTokenPruner._task_to_vision_attention(
            base, base_task, base_visual
        )
        padded_scores = AdaptiveVisualTokenPruner._task_to_vision_attention(
            padded, padded_task, padded_visual
        )
        assert torch.allclose(base_scores[0, 2:], padded_scores[0, 4:])
