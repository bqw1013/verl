# -*- coding:utf-8 -*-
"""
Author: Qiangwei Bai
Date: 2025-09-12 22:28:41
LastEditTime: 2025-09-16 19:59:03
LastEditors: Qiangwei Bai
FilePath: /verl/recipe/dare/core_algos.py
Description: 
"""
import torch
import numpy as np
import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import agg_loss
from collections import defaultdict
from typing import Optional


def compute_cto_gspo_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    relay_on_policy_mask,
    relay_off_policy_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "seq-mean-token-mean"
):
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    log_ratio_token = log_prob - old_log_prob
    log_ratio_token_clamped = torch.clamp(log_ratio_token, min=-10.0, max=10.0)
    ratio_token = torch.exp(log_ratio_token_clamped)

    log_dare_ratio_token = log_ratio_token_clamped - torch.log(ratio_token + 0.1)
    effective_log_ratio = torch.where(
        relay_off_policy_mask.bool(),
        log_dare_ratio_token,
        log_ratio_token,
    )

    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    seq_mean_log_ratio = torch.sum(effective_log_ratio * response_mask, dim=-1) / seq_lengths
    log_seq_importance_ratio = log_prob - log_prob.detach() + seq_mean_log_ratio.detach().unsqueeze(-1)
    ratio = torch.exp(log_seq_importance_ratio)

    relay_metrics = {}
    if relay_off_policy_mask.any():
        with torch.no_grad():
            raw_r = torch.exp(log_ratio_token)[relay_off_policy_mask.bool()]
            dare_r = torch.exp(log_dare_ratio_token)[relay_off_policy_mask.bool()]
            relay_metrics["relay/mean_raw_token_ratio"] = raw_r.mean().item()
            relay_metrics["relay/mean_dare_token_ratio"] = dare_r.mean().item()

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean")

    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    ppo_kl = verl_F.masked_mean(-log_ratio_token, response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, relay_metrics


def compute_dare_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    relay_on_policy_mask,
    relay_off_policy_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    raw_ratio = ratio.clone()

    # dare ratio
    relay_metrics = {}
    off_ratio = ratio / (ratio + 0.1)
    ratio = torch.where(relay_off_policy_mask.bool(), off_ratio, ratio)
    off_token_ratio = ratio[relay_off_policy_mask.bool()]
    raw_off_token_ratio = raw_ratio[relay_off_policy_mask.bool()]
    on_token_ratio = ratio[relay_on_policy_mask.bool()]
    if relay_on_policy_mask.any():
        relay_metrics["relay/max_on_policy_ratio"] = on_token_ratio.max().item()
        relay_metrics["relay/min_on_policy_ratio"] = on_token_ratio.min().item()
        relay_metrics["relay/mean_on_policy_ratio"] = on_token_ratio.mean().item()
    if relay_off_policy_mask.any():
        relay_metrics["relay/max_raw_off_policy_ratio"] = raw_off_token_ratio.max().item()
        relay_metrics["relay/min_raw_off_policy_ratio"] = raw_off_token_ratio.min().item()
        relay_metrics["relay/mean_raw_off_policy_ratio"] = raw_off_token_ratio.mean().item()
        relay_metrics["relay/max_off_policy_ratio"] = off_token_ratio.max().item()
        relay_metrics["relay/min_off_policy_ratio"] = off_token_ratio.min().item()
        relay_metrics["relay/mean_off_policy_ratio"] = off_token_ratio.mean().item()

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    if relay_off_policy_mask.any():
        off_policy_pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), relay_off_policy_mask)
        off_policy_pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), relay_off_policy_mask)
        relay_metrics["relay/off_policy_pg_clipfrac"] = off_policy_pg_clipfrac.item()
        relay_metrics["relay/off_policy_pg_clipfrac_lower"] = off_policy_pg_clipfrac_lower.item()

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, relay_metrics


def compute_grpo_outcome_advantage_v1(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                score = torch.sort(torch.tensor(id2score[idx]))[0]
                if score.sum() > 0 and score[:8].sum() == 0:
                    score = score[:8]
                    score[0] = 1
                else:
                    score = score[:8]
                id2mean[idx] = torch.mean(score)
                id2std[idx] = torch.std(score)
                # id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                # id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores
