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
Test training-only world token supervision in Gr00tN1d7ActionHead.

Adapted from GaussianDream++ (arXiv:2608.25659): World State / World
Prediction tokens join the DiT sequence and a training-only World
Representation Head adds auxiliary losses. These tests instantiate the action
head directly (no backbone required) and feed it synthetic backbone outputs,
mirroring test_action_head.py.
"""

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
import torch
from transformers.feature_extraction_utils import BatchFeature


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


def _make_backbone_output(config, batch_size=2, seq_len=8):
    return BatchFeature(
        data={
            "backbone_features": torch.randn(batch_size, seq_len, config.backbone_embedding_dim),
            "backbone_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "image_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
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


class TestWorldTokensDisabled:
    """Default config: no world tokens, forward/inference unchanged."""

    def test_no_world_head_by_default(self):
        head = Gr00tN1d7ActionHead(_small_config())
        assert head.world_head is None

    def test_forward_has_no_world_losses(self):
        config = _small_config()
        head = Gr00tN1d7ActionHead(config)
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert "loss" in out
        assert "world_state_loss" not in out
        assert "world_motion_loss" not in out


class TestWorldTokensEnabled:
    """Opt-in world token supervision."""

    def test_forward_returns_world_losses(self):
        config = _small_config(use_world_tokens=True)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert "world_state_loss" in out
        assert "world_motion_loss" in out
        assert torch.isfinite(out["loss"])
        assert torch.isfinite(out["world_state_loss"])
        assert torch.isfinite(out["world_motion_loss"])
        # Total loss must include the auxiliary terms.
        expected = (
            out["action_loss"].sum() / (out["action_mask"].sum() + 1e-6)
            + config.world_state_loss_weight * out["world_state_loss"]
            + config.world_motion_loss_weight * out["world_motion_loss"]
        )
        assert torch.allclose(out["loss"], expected)

    def test_world_head_receives_gradients(self):
        config = _small_config(use_world_tokens=True)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        out["loss"].backward()
        assert head.world_head.state_tokens.grad is not None
        assert head.world_head.prediction_tokens.grad is not None
        assert torch.isfinite(head.world_head.state_tokens.grad).all()

    def test_action_slice_unaffected_by_prepended_tokens(self):
        config = _small_config(use_world_tokens=True)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert out["action_loss"].shape == (2, config.action_horizon, config.max_action_dim)

    def test_get_action_with_world_tokens(self):
        """Inference keeps the tokens but drops the decode head and losses."""
        config = _small_config(use_world_tokens=True)
        head = Gr00tN1d7ActionHead(config)
        head.eval()
        action_input = _make_action_input(config)
        del action_input["action"]
        out = head.get_action(_make_backbone_output(config), action_input)
        assert out["action_pred"].shape == (2, config.action_horizon, config.max_action_dim)
        assert not out["action_pred"].requires_grad

    def test_zero_loss_weights_recover_action_only_loss(self):
        config = _small_config(
            use_world_tokens=True,
            world_state_loss_weight=0.0,
            world_motion_loss_weight=0.0,
        )
        head = Gr00tN1d7ActionHead(config)
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        action_only = out["action_loss"].sum() / (out["action_mask"].sum() + 1e-6)
        assert torch.allclose(out["loss"], action_only)
