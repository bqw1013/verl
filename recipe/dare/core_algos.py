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
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss


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

    # dare ratio
    relay_metrics = {}
    off_ratio = ratio / (ratio + 0.1)
    ratio = torch.where(relay_off_policy_mask.bool(), off_ratio, ratio)
    off_token_ratio = ratio[relay_off_policy_mask.bool()]
    on_token_ratio = ratio[relay_on_policy_mask.bool()]
    if relay_on_policy_mask.any():
        relay_metrics["relay/max_on_policy_ratio"] = on_token_ratio.max().item()
        relay_metrics["relay/min_on_policy_ratio"] = on_token_ratio.min().item()
        relay_metrics["relay/mean_on_policy_ratio"] = on_token_ratio.mean().item()
    if relay_off_policy_mask.any():
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