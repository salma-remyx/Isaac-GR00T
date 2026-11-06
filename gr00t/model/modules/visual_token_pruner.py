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

"""Task-conditioned, temporally adaptive visual token pruning for the action head.

Camera images expand into hundreds-to-thousands of visual tokens that dominate
the action head's cross-attention cost at every closed-loop control step. This
module implements the training-free pruning mechanism of FastVLA (Xu et al.,
"FastVLA: Task-Conditioned and Temporally Adaptive Visual Token Compression
for Efficient Vision-Language-Action Models", arXiv:2610.22146), adapted to the
GR00T N1.7 backbone feature stream:

1. **Task-conditioned importance.** FastVLA scores each visual token by the
   attention it receives from the task tokens, read from the LLM's attention
   maps. The frozen Qwen3-VL backbone does not expose those maps at the action
   head call site, so the same task-to-vision attention is computed here as a
   parameter-free single head over the backbone features: a scaled dot product
   with a shared (identity) query/key projection, softmaxed over visual tokens
   and summed over task tokens. Tokens describing the current goal thereby
   vote for the visual tokens they relate to — no learned estimator, no
   training, and importance tracks the task instead of generic saliency.
2. **Temporally adaptive budget.** The keep ratio is set per step from the
   uniformity of the importance distribution via the normalized inverse-Simpson
   index (``1 / sum(p^2) / N``): a peaked distribution means a few tokens carry
   the task and the tail can be dropped aggressively, while a near-uniform
   distribution means control-relevant evidence is spread across the visual
   field and pruning must stay conservative. The current uniformity is
   calibrated against a sliding window of recent steps, so budgets follow scene
   dynamics rather than a static threshold.

Pruning compacts the kept tokens to the front of the sequence and rewrites
``backbone_features`` / ``backbone_attention_mask`` / ``image_mask`` in place.
Every downstream consumer of the visual stream (``vl_self_attention``, the DiT's
``encoder_hidden_states`` + image/text attention-mask path) is mask-driven and
sequence-length agnostic, so the pruned batch flows through unchanged code
paths at a shorter sequence length.

The pruner is opt-in via the ``visual_token_pruner_cfg`` config key (see
``Gr00tN1d7ActionHead``). It only runs in eval mode by default, matching
FastVLA's training-free, inference-time design.
"""

from collections import deque
from dataclasses import dataclass
import logging
import math

import torch
from transformers.feature_extraction_utils import BatchFeature


logger = logging.getLogger(__name__)


@dataclass
class PruningStats:
    """Per-step record of the adaptive pruning decision (for logging and tests)."""

    uniformity: float  # normalized inverse-Simpson index of visual importance, in (0, 1]
    keep_ratio: float  # calibrated visual keep ratio for this step, in [min, max]
    in_tokens: int  # full stream length before pruning (task + visual tokens)
    out_tokens: int  # full stream length after pruning


class AdaptiveVisualTokenPruner:
    """Training-free visual token pruner for a mixed vision-language stream.

    On each call, scores visual tokens by task-to-vision attention, sets the
    per-step visual token budget from the importance distribution's uniformity
    (calibrated against a sliding window of recent steps), and compacts the
    surviving tokens to the front of ``backbone_output``.

    Args:
        min_keep_ratio: visual keep ratio for the most peaked importance
            distributions seen by the calibration window.
        max_keep_ratio: visual keep ratio for the most uniform importance
            distributions seen by the calibration window.
        min_visual_tokens: floor on kept visual tokens per sample, so a very
            peaked distribution can never strip the visual stream bare.
        window_size: number of recent steps used to calibrate the
            uniformity-to-ratio mapping.
        apply_during_training: also prune on the training forward pass. FastVLA
            prunes at inference only, and the sliding-window calibration assumes
            temporally consecutive control steps, so this defaults to False.
    """

    def __init__(
        self,
        min_keep_ratio: float = 0.25,
        max_keep_ratio: float = 0.75,
        min_visual_tokens: int = 16,
        window_size: int = 8,
        apply_during_training: bool = False,
    ):
        if not 0.0 < min_keep_ratio <= max_keep_ratio <= 1.0:
            raise ValueError(
                f"Require 0 < min_keep_ratio <= max_keep_ratio <= 1, "
                f"got min_keep_ratio={min_keep_ratio}, max_keep_ratio={max_keep_ratio}"
            )
        if min_visual_tokens < 1:
            raise ValueError(f"min_visual_tokens must be >= 1, got {min_visual_tokens}")
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {window_size}")

        self.min_keep_ratio = min_keep_ratio
        self.max_keep_ratio = max_keep_ratio
        self.min_visual_tokens = min_visual_tokens
        self.apply_during_training = apply_during_training
        self.last_stats: PruningStats | None = None
        self._window: deque[float] = deque(maxlen=window_size)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(min_keep_ratio={self.min_keep_ratio}, "
            f"max_keep_ratio={self.max_keep_ratio}, min_visual_tokens={self.min_visual_tokens}, "
            f"window_size={self._window.maxlen}, "
            f"apply_during_training={self.apply_during_training})"
        )

    def reset(self) -> None:
        """Clear the sliding-window calibration state (e.g. between episodes)."""
        self._window.clear()
        self.last_stats = None

    def __call__(self, backbone_output: BatchFeature) -> BatchFeature:
        """Prune the visual stream of ``backbone_output`` in place and return it."""
        features = backbone_output["backbone_features"]
        attention_mask = backbone_output["backbone_attention_mask"].to(torch.bool)
        image_mask = backbone_output.get("image_mask")
        if image_mask is None:
            # Without an image mask the visual stream cannot be identified, so
            # pruning is a no-op (the non-alternate DiT path never sets one).
            return backbone_output
        image_mask = image_mask.to(torch.bool)

        task_mask = attention_mask & ~image_mask
        visual_mask = attention_mask & image_mask
        num_task = task_mask.sum(dim=1)
        num_visual = visual_mask.sum(dim=1)
        if bool((num_task == 0).any()) or bool((num_visual == 0).any()):
            # A sample without task or visual tokens has no pruning signal;
            # keep the unpruned stream for the whole batch.
            return backbone_output

        importance = self._task_to_vision_attention(features, task_mask, visual_mask)
        keep_count, uniformity, keep_ratio = self._budget(num_visual, importance)

        # Rank positions by importance (visual tokens first, -inf elsewhere) and
        # keep each sample's top ``keep_count`` visual tokens. Two stable
        # argsorts give exact ranks, with deterministic tie-breaking by position.
        order = importance.argsort(dim=1, descending=True, stable=True)
        rank = order.argsort(dim=1, stable=True)
        keep_visual = (rank < keep_count.unsqueeze(1)) & visual_mask

        keep = task_mask | keep_visual
        new_len = self._compact(backbone_output, keep)
        self.last_stats = PruningStats(
            uniformity=uniformity,
            keep_ratio=keep_ratio,
            in_tokens=int(features.shape[1]),
            out_tokens=new_len,
        )
        logger.debug(
            f"Visual token pruning: uniformity={uniformity:.3f} "
            f"keep_ratio={keep_ratio:.3f} tokens {features.shape[1]} -> {new_len}"
        )
        return backbone_output

    @staticmethod
    def _task_to_vision_attention(
        features: torch.Tensor, task_mask: torch.Tensor, visual_mask: torch.Tensor
    ) -> torch.Tensor:
        """Importance of each position: attention received from task tokens.

        Parameter-free single-head attention with a shared (identity) query/key
        projection over the backbone features: for every task token, softmax
        over visual tokens of the scaled dot product, then sum over task tokens.
        Returns ``[B, S]`` with non-negative scores supported on visual
        positions (FastVLA reads the equivalent quantity from the LLM's
        attention maps; see the module docstring for the substitution).
        """
        feats = features.detach().to(torch.float32)
        scale = math.sqrt(feats.shape[-1])
        logits = torch.bmm(feats, feats.transpose(1, 2)) / scale
        logits = logits.masked_fill(~visual_mask[:, None, :], float("-inf"))
        task_to_vision = torch.softmax(logits, dim=-1)
        task_to_vision = task_to_vision * task_mask[:, :, None].to(feats.dtype)
        return task_to_vision.sum(dim=1)

    def _budget(
        self, num_visual: torch.Tensor, importance: torch.Tensor
    ) -> tuple[torch.Tensor, float, float]:
        """Per-sample number of visual tokens to keep, from importance uniformity."""
        probs = importance / importance.sum(dim=1, keepdim=True).clamp_min(1e-12)
        simpson = (probs * probs).sum(dim=1)
        # Inverse-Simpson "effective token count" in [1, num_visual], normalized
        # to (0, 1] so the metric is comparable across visual token counts.
        uniformity = (1.0 / simpson.clamp_min(1e-12)) / num_visual.clamp_min(1)
        uniformity = float(uniformity.mean())

        # Calibrate against recent steps: a step flatter than the recent window
        # keeps more visual tokens, a more peaked one keeps fewer. Without
        # contrast (first step, static scene) use the midpoint budget.
        self._window.append(uniformity)
        lo, hi = min(self._window), max(self._window)
        alpha = 0.5 if hi - lo < 1e-6 else (uniformity - lo) / (hi - lo)
        keep_ratio = self.min_keep_ratio + alpha * (self.max_keep_ratio - self.min_keep_ratio)

        keep_count = torch.ceil(keep_ratio * num_visual.to(torch.float32)).to(torch.int64)
        keep_count = keep_count.clamp(min=self.min_visual_tokens)
        keep_count = torch.minimum(keep_count, num_visual)
        return keep_count, uniformity, keep_ratio

    @staticmethod
    def _compact(backbone_output: BatchFeature, keep: torch.Tensor) -> int:
        """Compact kept tokens to the front of the sequence, masking dropped tail.

        Rewrites ``backbone_features`` (dropped positions zeroed),
        ``backbone_attention_mask`` and ``image_mask`` in ``backbone_output`` so
        every downstream consumer sees a consistent, shorter stream. Returns the
        new sequence length.
        """
        features = backbone_output["backbone_features"]
        attention_mask = backbone_output["backbone_attention_mask"].to(torch.bool)
        image_mask = backbone_output["image_mask"].to(torch.bool)

        num_keep = keep.sum(dim=1)
        new_len = int(num_keep.max())
        # Stable sort on the keep mask: kept tokens first, original order preserved.
        order = keep.to(torch.int64).argsort(dim=1, descending=True, stable=True)
        gather_idx = order[:, :new_len]

        gather_feats = gather_idx.unsqueeze(-1).expand(-1, -1, features.shape[-1])
        pruned_features = features.gather(1, gather_feats)
        pruned_attention = attention_mask.gather(1, gather_idx)
        pruned_image = image_mask.gather(1, gather_idx)

        # Rows shorter than new_len pad the tail; those positions must be masked
        # out and zeroed so no consumer attends stale duplicates of dropped tokens.
        tail_valid = torch.arange(new_len, device=keep.device)[None, :] < num_keep[:, None]
        backbone_output["backbone_features"] = pruned_features * tail_valid.unsqueeze(-1).to(
            pruned_features.dtype
        )
        backbone_output["backbone_attention_mask"] = pruned_attention & tail_valid
        backbone_output["image_mask"] = pruned_image & tail_valid
        return new_len
