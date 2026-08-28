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

"""Training-only world-modeling token supervision for the GR00T action head.

Adapted from GaussianDream++ (arXiv:2608.25659): dedicated World State Tokens
and World Prediction Tokens are inserted into the policy transformer's token
sequence, and a training-only World Representation Head decodes them into a
current-world reconstruction and a coupled future-motion prediction. The
static/dynamic factorization keeps persistent scene structure on the state
tokens and residual motion on the prediction tokens. At inference the decode
head and auxiliary losses are dropped; only the world tokens remain in the
sequence, so the policy keeps its trained behavior at negligible extra cost.

Target-native substitutions relative to the paper:
  - Gaussian-primitive decoding + differentiable splat rendering are replaced
    by lightweight MLP decoders (the repo has no Gaussian renderer).
  - VGGT/TGE 3D targets and future RGB frames (not present in GR00T training
    batches) are replaced by the current proprioceptive state (persistent
    structure proxy) and the future relative-EEF action chunk (short-horizon
    dynamics proxy — relative actions already encode residual motion).
"""

import torch
from torch import nn
import torch.nn.functional as F


class WorldTokenHead(nn.Module):
    """World State / World Prediction tokens plus their training-only decoders."""

    def __init__(
        self,
        num_state_tokens: int,
        num_prediction_tokens: int,
        token_dim: int,
        decode_dim: int,
        state_dim: int,
        action_horizon: int,
        action_dim: int,
    ):
        """
        Args:
            num_state_tokens: Number of World State Tokens (persistent structure).
            num_prediction_tokens: Number of World Prediction Tokens (residual motion).
            token_dim: Dimension of the transformer sequence the tokens join.
            decode_dim: Dimension of the transformer outputs the decoders read.
            state_dim: Dimension of the proprioceptive state target.
            action_horizon: Number of future action steps to predict.
            action_dim: Dimension of each action step.
        """
        super().__init__()
        self.num_state_tokens = num_state_tokens
        self.num_prediction_tokens = num_prediction_tokens
        self.action_horizon = action_horizon
        self.action_dim = action_dim

        self.state_tokens = nn.Parameter(torch.randn(1, num_state_tokens, token_dim) * 0.02)
        self.prediction_tokens = nn.Parameter(
            torch.randn(1, num_prediction_tokens, token_dim) * 0.02
        )

        self.state_decoder = nn.Sequential(
            nn.Linear(decode_dim, decode_dim),
            nn.GELU(),
            nn.Linear(decode_dim, state_dim),
        )
        self.motion_decoder = nn.Sequential(
            nn.Linear(decode_dim, decode_dim),
            nn.GELU(),
            nn.Linear(decode_dim, action_horizon * action_dim),
        )

    @property
    def num_tokens(self) -> int:
        return self.num_state_tokens + self.num_prediction_tokens

    def append_to_sequence(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Prepend the world tokens to a token sequence.

        Args:
            hidden_states: [B, S, token_dim] token sequence.

        Returns:
            [B, num_tokens + S, token_dim] sequence with world tokens first, so
            downstream slices of the trailing state/action tokens are unaffected.
        """
        batch_size = hidden_states.shape[0]
        tokens = torch.cat((self.state_tokens, self.prediction_tokens), dim=1)
        tokens = tokens.expand(batch_size, -1, -1).to(hidden_states.dtype)
        return torch.cat((tokens, hidden_states), dim=1)

    def compute_losses(
        self,
        world_token_output: torch.Tensor,
        state_target: torch.Tensor,
        motion_target: torch.Tensor,
        motion_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Decode world token outputs and compute the auxiliary losses.

        Args:
            world_token_output: [B, num_tokens, decode_dim] transformer outputs at
                the world-token positions.
            state_target: [B, state_dim] current proprioceptive state.
            motion_target: [B, action_horizon, action_dim] future action chunk.
            motion_mask: [B, action_horizon, action_dim] valid-action mask.

        Returns:
            Dict with scalar ``world_state_loss`` and ``world_motion_loss``.
        """
        state_out = world_token_output[:, : self.num_state_tokens]
        prediction_out = world_token_output[:, self.num_state_tokens :]

        pred_state = self.state_decoder(state_out.mean(dim=1))
        world_state_loss = F.mse_loss(pred_state.float(), state_target.float())

        pred_motion = self.motion_decoder(prediction_out.mean(dim=1))
        pred_motion = pred_motion.view(-1, self.action_horizon, self.action_dim)
        motion_loss = (
            F.mse_loss(pred_motion.float(), motion_target.float(), reduction="none") * motion_mask
        )
        world_motion_loss = motion_loss.sum() / (motion_mask.sum() + 1e-6)

        return {
            "world_state_loss": world_state_loss,
            "world_motion_loss": world_motion_loss,
        }
