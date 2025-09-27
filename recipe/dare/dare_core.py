# -*- coding:utf-8 -*-
import torch
import random
import itertools
import numpy as np
import scipy.stats as stats

from collections import defaultdict
from typing import Union, List, Callable, Tuple
from scipy.special import softmax
from transformers import PreTrainedTokenizer
from torch.nn.utils.rnn import pad_sequence
from verl.utils.torch_functional import pad_sequence_to_length
from verl import DataProto


def cosine_annealing(t_train: int, total_decay_steps: int, initial_alpha: float = 1.0, final_alpha: float = 0.0) -> float:
    """
    Calculates the relay participation rate `alpha_t` using cosine annealing schedule.

    The cosine schedule provides a smooth, non-linear decay that is slow at the beginning and end of 
    schedule, and fast in the middle, which can be beneficial for stable learning.

    Args:
        t_train: The current training step.
        total_decay_steps: The total number of training steps.
        initial_alpha: The starting value of alpha at t_train = 0. Represents high participation rate.
        final_alpha: The final value of alpha at t_train = total_decay_steps. Represents low participation rate.

    Returns:
        The relay participation rate `alpha_t` at the current training step.
    """
    if t_train > total_decay_steps:
        return final_alpha

    cosine_decay = 0.5 * (1 + np.cos(np.pi * t_train / total_decay_steps))
    return final_alpha + (initial_alpha - final_alpha) * cosine_decay


def get_global_preference(response_mask: torch.Tensor, alpha_t: float, std_ratio: float = 0.1) -> torch.Tensor:
    """
    Generates a global preference distribution for the handover point for a batch of trajectories.

    Args:
        response_mask: A boolean tensor of shape `(batch_size, max_length)` where `True` indicates a valid token 
            and `False` indicates a padding token.
            Example: `tensor([[True, True, True, False],
                             [True, True, False, False]])`
        alpha_t: The current relay participation rate, from a cosine annealing schedule. A single float applied
            to all trajectories in the batch.
        std_ratio: The standard deviation ratio, applied to all trajectories in the batch.

    Returns:
        torch.Tensor: A float tensor of shape `(batch_size, max_length)` where each row contains the preference 
            distribution for the handover point for the corresponding trajectory. Probabilities for padding tokens 
            are zero.
            Example: For the mask above and alpha_t=0.5, the output might be 
                `tensor([[0.3, 0.4, 0.3, 0.0],
                         [0.5, 0.5, 0.0, 0.0]])`
    """
    Ls = response_mask.sum(-1)
    Ls = Ls.cpu().numpy()
    batch_size = response_mask.shape[0]
    max_length = response_mask.shape[1]

    global_prefs = torch.zeros_like(response_mask, dtype=torch.float32)

    for i in range(batch_size):
        L = int(Ls[i])
        if L <= 1:
            global_prefs[i, 0] = 1.0
            continue

        mu = L * (1 - alpha_t)
        sigma = max(L * std_ratio, 1.0)

        x_domain = np.arange(1, L + 1)
        a, b = (1 - mu)/sigma, (L - mu)/sigma
        dist = stats.truncnorm(a, b, loc=mu, scale=sigma)

        pdf_values = dist.pdf(x_domain)
        normalized = pdf_values / np.sum(pdf_values)
        global_prefs[i, :L] = torch.from_numpy(normalized)

    return global_prefs


def get_local_attraction(entropies: torch.Tensor, response_mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Computes local attraction probabilities for a batch of entropy sequences.

    Args:
        entropies: A float tensor of shape `(batch_size, max_length)` where each row contains the entropy 
            values for the corresponding trajectory. Values in padding positions can by anything as they 
            will be masked out.
            Example: `tensor([[0.2, 1.5, 0.8, 0.0],
                             [0.5, 2.0, 0.0, 0.0]])`
        response_mask: A boolean tensor of shape `(batch_size, max_length)` where `True` indicates a valid token 
            and `False` indicates a padding token.
            Example: `tensor([[True, True, True, False],
                             [True, True, False, False]])`
        temperature: The softmax temperature. Controls the sharpness of the distribution.

    Returns:
        torch.Tensor: A float tensor of shape `(batch_size, max_length)` where each row is the local attraction 
            probability distribution for the corresponding trajectory. Probabilities for padding tokens are zero.
            Example: For the inputs above, the output might be
                `tensor([[0.09, 0.67, 0.24, 0.00],
                         [0.18, 0.82, 0.00, 0.00]])`
    """
    device = entropies.device
    tau = max(temperature, 1e-6)

    if response_mask.dtype != torch.bool:
        response_mask = response_mask.bool()

    scaled_entropy = entropies / tau
    scaled_entropy = torch.where(
        response_mask,
        scaled_entropy,
        torch.tensor(-torch.inf, device=device)
    )

    return torch.softmax(scaled_entropy, dim=-1)


def sample_relay_point(
    entropies: torch.Tensor,
    response_mask: torch.Tensor, 
    t_train: int, 
    total_decay_steps: int,
    std_ratio: float = 0.1,
    temperature: float = 1.0) -> torch.Tensor:
    """
    Samples a relay point for a batch of trajectories.
    """
    if entropies.shape != response_mask.shape:
        raise ValueError("entropies and response_mask must have the same shape")
    
    device = entropies.device

    alpha_t = cosine_annealing(
        t_train, 
        total_decay_steps,
        initial_alpha=1.0,
        final_alpha=0.0,
    )
    p_global_batch = get_global_preference(response_mask, alpha_t, std_ratio).to(device)
    p_local_batch = get_local_attraction(entropies, response_mask, temperature)
    
    final_scores = p_global_batch * p_local_batch
    sum_scores = final_scores.sum(dim=-1, keepdim=True)

    final_probs = torch.where(
        sum_scores > 1e-9,
        final_scores / sum_scores,
        p_global_batch,
    )
    
    sampled_indices = torch.multinomial(final_probs, num_samples=1).squeeze(-1)
    relay_points = sampled_indices
    # print(f"t_train: {t_train}, alpha_t: {alpha_t}, global_prefs: {torch.argmax(p_global_batch, dim=1).float().mean()}, relay_points: {relay_points.float().mean()}")
    return relay_points


def combine_prompt_response_by_relay_point(
    prompts: torch.Tensor,
    responses: torch.Tensor,
    relay_points: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
) -> Union[torch.Tensor, List[List[int]]]:
    """
    Combines the prompt and response into a single tensor, with the relay point as the separator.
    """
    if prompts.shape[0] != responses.shape[0] or prompts.shape[0] != relay_points.shape[0]:
        raise ValueError("prompts, responses and relay_points must have the same batch size")

    pad_token_id = tokenizer.pad_token_id
    batch_size = prompts.shape[0]
    prompt_lengths = (prompts != pad_token_id).sum(dim=-1)

    relay_prompts = []
    for i in range(batch_size):
        p_len = prompt_lengths[i].item()
        r_len = relay_points[i].item()

        actual_prompt = prompts[i, -p_len:] if p_len > 0 else torch.tensor([], dtype=prompts.dtype, device=prompts.device)
        response_part = responses[i, :r_len] if r_len > 0 else torch.tensor([], dtype=responses.dtype, device=responses.device)
        combined = torch.cat([actual_prompt, response_part], dim=-1)
        relay_prompts.append(combined)
    
    # relay_prompts_non_tensor = [prompt.tolist() for prompt in relay_prompts]

    max_new_len = max(len(seq) for seq in relay_prompts)
    relay_prompts = [pad_sequence_to_length(seq, max_new_len, pad_token_id, left_pad=True) for seq in relay_prompts]
    relay_prompts = torch.stack(relay_prompts, dim=0)
    
    return relay_prompts

def determine_relay_count(c: int, n: int, k_max: int, beta: float = 1.0) -> int:
    """
    Determing the total number of trajectories to be relayed.

    The number of relays, `k`, is sampled from a Binomial distribution, making the process probabilistic and smooth.
    The expected value of `k` is designed to decrease as the success rate increases.

    Args:
        c: The number of successful trajectories (rollouts) out of `n`.
        n: The total number of trajectories.
        k_max: The maximum nuber of trajectories that can be selected for argumentation.
        beta: A coefficient that controls the steepness of the decay curve for the expected number of relays.

    Returns:
        int: The sampled number of trajectories to be relayed. This is an integer between 0 and `n`.
    """
    if n <= 0:
        return 0
    if not (0 <= c <= n):
        raise ValueError(f"c must be between 0 and n, but got {c} and {n}")

    p = c / n

    expected_k = k_max * ((1 - p) ** beta)

    p_k = np.clip(expected_k / n, 0, 1)

    k = np.random.binomial(n, p_k)

    return k

def allocate_relay_budget(k: int, c: int, n: int, gamma: float = 1.0) -> Tuple[int, int]:
    """
    Allocates the total augmentation budget between failed and successful trajectories.

    The allocation is probabilistic, governed by a preference for augmenting failed trajectories, 
    especially when the success rate `c / n` is low. This preference is controlled by the parameter `gamma`.

    Args:
        k: The total number of trajectories to be relayed.
        c: The number of successful trajectories.
        n: The total number of trajectories.
        gamma: The preference for augmenting failed trajectories.

    Returns:
        Tuple[int, int]: A tuple containing the number of slots allocated to successful trajectories (`k_succ`) 
        and failed trajectories (`k_fail`).
    """
    if k == 0:
        return 0, 0
    if not (0 <= c <= n and n > 0):
        raise ValueError(f"Success count `c` must be between 0 and `n`, and `n` must be positive.")
    
    p = c / n
    p_fail = (1 - p) ** gamma
    p_fail = np.clip(p_fail, 0, 1)

    k_succ, k_fail = 0, 0
    available_succ, available_fail = c, n - c

    for _ in range(k):
        if available_succ == 0 and available_fail == 0:
            break

        choice = np.random.choice(["fail", "succ"], p=[p_fail, 1 - p_fail])
        if choice == "fail":
            if available_fail > 0:
                k_fail += 1
                available_fail -= 1
            else:
                k_succ += 1
                available_succ -= 1
        else:
            if available_succ > 0:
                k_succ += 1
                available_succ -= 1
            else:
                k_fail += 1
                available_fail -= 1

    return k_succ, k_fail

def determine_relay_samples(rewards: torch.Tensor, uids: List[int], relay_sample_fn: Callable = None):
    if rewards.shape[0] != len(uids):
        raise ValueError("rewards and uid must have the same length")
    
    id2index_reward = defaultdict(list)
    for index, uid in enumerate(uids):
        id2index_reward[uid].append((index, rewards[index]))
    
    beta = 0.5
    gamma = 2.0
    sampled_succ_count = 0
    sampled_fail_count = 0
    sampled_indexs = []
    for uid, index_rewards in id2index_reward.items():
        group_indexs = [index_reward[0] for index_reward in index_rewards]
        group_rewards = [index_reward[1] for index_reward in index_rewards]

        n = len(group_indexs)
        c = int(sum(group_rewards).item())
        k_max = max(int(n/2), 1)
        k = determine_relay_count(c, n, k_max, beta)
        k_succ, k_fail = allocate_relay_budget(k, c, n, gamma)
        sampled_succ_count += k_succ
        sampled_fail_count += k_fail
        succ_indexs = [idx for idx, reward in zip(group_indexs, group_rewards) if reward == 1]
        fail_indexs = [idx for idx, reward in zip(group_indexs, group_rewards) if reward == 0]
        sampled_succ_indexs = np.random.choice(succ_indexs, k_succ, replace=False).tolist()
        sampled_fail_indexs = np.random.choice(fail_indexs, k_fail, replace=False).tolist()
        sampled_indexs.extend(sampled_succ_indexs + sampled_fail_indexs)

    sampled_indexs.sort()
    sampled_mask = torch.zeros(rewards.shape[0], dtype=torch.bool)
    sampled_mask[sampled_indexs] = True

    return sampled_mask, sampled_succ_count, sampled_fail_count


def compute_relay_reward(
    data_source: List[str], 
    solution_str: List[str], 
    ground_truth: List[str], 
    extra_info: List[str], 
    compute_score: Callable
) -> List[float]:
    relay_reward = []
    for data_source, solution_str, ground_truth, extra_info in zip(data_source, solution_str, ground_truth, extra_info):
        score = compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info)
        relay_reward.append(score)

    return relay_reward


def update_batch(
    batch: DataProto,
    relay_responses: List[List[int]],
    relay_logprobs: List[List[float]],
    relay_reward: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
):
    relay_responses = list(itertools.compress(relay_responses, relay_reward))
    relay_logprobs = list(itertools.compress(relay_logprobs, relay_reward))

    # update relay_samples_mask
    relay_samples_mask = batch.batch["relay_samples_mask"]
    relay_pos_index = relay_samples_mask.nonzero().squeeze()[relay_reward.bool()]
    relay_samples_mask = torch.zeros_like(relay_samples_mask)
    relay_samples_mask[relay_pos_index] = True
    batch.batch["relay_samples_mask"] = relay_samples_mask

    # get raw responses, old_log_probs and relay_points
    relay_samples_raw_responses = batch.batch["responses"][relay_samples_mask]
    relay_samples_raw_old_logprobs = batch.batch["old_log_probs"][relay_samples_mask]
    relay_samples_relay_points = batch.batch["relay_points"][relay_samples_mask]

    # mix relay_responses and relay_logprobs
    mixed_responses = []
    mixed_logprobs = []

    for i in range(relay_samples_raw_responses.shape[0]):
        raw_response = relay_samples_raw_responses[i]
        raw_old_logprob = relay_samples_raw_old_logprobs[i]
        relay_point = relay_samples_relay_points[i]
        relay_response = torch.tensor(relay_responses[i], dtype=raw_response.dtype, device=raw_response.device)
        relay_logprob = torch.tensor(relay_logprobs[i], dtype=raw_old_logprob.dtype, device=raw_old_logprob.device)
        mixed_response = torch.cat([raw_response[:relay_point], relay_response])
        mixed_logprob = torch.cat([raw_old_logprob[:relay_point], relay_logprob])
        mixed_responses.append(mixed_response)
        mixed_logprobs.append(mixed_logprob)
    assert all([len(mixed_responses[i])==len(mixed_logprobs[i]) for i in range(len(mixed_responses))]), "mixed_responses and mixed_logprobs must have the same length"
    
    # pad mixed_responses and mixed_logprobs
    max_response_length = batch.batch["responses"].shape[1]
    mixed_responses = pad_sequence(mixed_responses, batch_first=True, padding_value=tokenizer.pad_token_id)
    mixed_responses = pad_sequence_to_length(mixed_responses, max_response_length, tokenizer.pad_token_id, left_pad=False)
    mixed_logprobs = pad_sequence(mixed_logprobs, batch_first=True, padding_value=0)
    mixed_logprobs = pad_sequence_to_length(mixed_logprobs, max_response_length, 0, left_pad=False)
    mixed_response_mask = (mixed_responses!=tokenizer.pad_token_id).long()

    # update batch
    batch.batch["raw_responses"] = batch.batch["responses"].detach().clone()
    batch.batch["raw_old_log_probs"] = batch.batch["old_log_probs"].detach().clone()
    batch.batch["raw_response_mask"] = batch.batch["response_mask"].detach().clone()
    batch.batch["raw_token_level_scores"] = batch.batch["token_level_scores"].detach().clone()

    batch.batch["responses"][relay_samples_mask] = mixed_responses
    batch.batch["old_log_probs"][relay_samples_mask] = mixed_logprobs
    batch.batch["response_mask"][relay_samples_mask] = mixed_response_mask

    # assert batch.batch["token_level_scores"][relay_samples_mask].sum()==0
    batch.batch["token_level_scores"][relay_samples_mask] = 0
    batch.batch["token_level_scores"][relay_samples_mask, mixed_response_mask.sum(-1) - 1] = 1
    # assert (batch.batch["responses"][~relay_samples_mask] == batch.batch["raw_responses"][~relay_samples_mask]).all()
    # assert (batch.batch["old_log_probs"][~relay_samples_mask] == batch.batch["raw_old_log_probs"][~relay_samples_mask]).all()
    # batch.batch["token_level_scores"][torch.arange(len(batch.batch["response_mask"])), batch.batch["response_mask"].sum(dim=-1)-1]
    return batch

def test_global_and_local_batch():
    batch_size = 4
    max_length = 10
    lengths = torch.tensor([5, 8, 10, 4])
    response_mask = torch.arange(max_length).expand(batch_size, -1) < lengths.unsqueeze(1)
    entropies = torch.randn(batch_size, max_length) * response_mask.float()

    # 测试global batch
    alpha_t = 0.5
    batch_result_global = get_global_preference_batch(response_mask, alpha_t)
    manual_result_global = torch.zeros_like(batch_result_global)
    for i in range(batch_size):
        L = lengths[i].item()
        res = get_global_preference(L, alpha_t)
        manual_result_global[i, :L] = torch.from_numpy(res)

    assert torch.allclose(batch_result_global, manual_result_global), "get_global_preference_batch 测试失败!"

    # 测试local batch
    temperature = 0.8
    batch_result_local = get_local_attraction_batch(entropies, response_mask, temperature)
    manual_result_local = torch.zeros_like(batch_result_local)
    for i in range(batch_size):
        L = lengths[i].item()
        valid_entropies = entropies[i, :L].numpy()
        res = get_local_attraction(valid_entropies, temperature)
        manual_result_local[i, :L] = torch.from_numpy(res)

    assert torch.allclose(batch_result_local, manual_result_local), "get_local_attraction_batch 测试失败!"

def test_determine_relay_count():
    n = 8
    beta = 0.5
    k_max = 4
    for c in range(n + 1):
        k_list = []
        for _ in range(1000):
            k = determine_relay_count(c, n, k_max, beta)
            k_list.append(k)
        print(f"c: {c}, average k: {np.mean(k_list)}")

def test_allocate_relay_budget():
    n = 8
    gamma = 2.0
    for k in range(n + 1):
        for c in range(n + 1):
            k_succ_list = []
            k_fail_list = []
            for _ in range(1000):
                k_succ, k_fail = allocate_relay_budget(k, c, n, gamma)
                k_succ_list.append(k_succ)
                k_fail_list.append(k_fail)
            print(f"k: {k}, c: {c}, k_succ: {np.mean(k_succ_list)}, k_fail: {np.mean(k_fail_list)}")

if __name__ == "__main__":
    # test_determine_relay_count()
    test_allocate_relay_budget()
    pass
