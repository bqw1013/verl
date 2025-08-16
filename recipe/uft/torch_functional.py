import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

def pad_sequence_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
    """
    pad a 2D tensors (e.g. responses, logprobs) in the last dim to max_seq_length.
    input shape: [bs, seq_length]
    output shape: [bs, max_seq_length]
    """
    if tensors.shape[-1] >= max_seq_len:
        return tensors
    # (0, max_seq_len - tensors.shape[-1]) means right pad to max_seq_length and no left pad
    pad_tuple = (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
    return F.pad(tensors, pad_tuple, "constant", pad_token_id)

def postprocess_data(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_length: int,
    pad_token_id: int,
    left_pad=True,
    truncation="error",
):
    """Process tokenizer outputs to consistent shapes via padding/truncation.

    Args:
        input_ids: Token indices [batch_size, seq_len]
        attention_mask: Mask [batch_size, seq_len]
        max_length: Target sequence length
        pad_token_id: Padding token ID
        left_pad: Pad left if True
        truncation: "left", "right", "middle" or "error"

    Returns:
        (input_ids, attention_mask) padded/truncated to max_length
    """
    assert truncation in ["left", "right", "middle", "error"]
    assert input_ids.ndim == 2

    sequence_length = input_ids.shape[-1]
    if sequence_length < max_length:
        input_ids = pad_sequence_to_length(
            input_ids, max_seq_len=max_length, pad_token_id=pad_token_id, left_pad=left_pad
        )
        attention_mask = pad_sequence_to_length(
            attention_mask, max_seq_len=max_length, pad_token_id=0, left_pad=left_pad
        )
    elif sequence_length > max_length:
        if truncation == "left":
            # actually, left truncation may not be reasonable
            input_ids = input_ids[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]
        elif truncation == "right":
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
        elif truncation == "middle":
            left_half = max_length // 2
            right_half = max_length - left_half
            input_ids = torch.cat([input_ids[:, :left_half], input_ids[:, -right_half:]], dim=-1)
            attention_mask = torch.cat([attention_mask[:, :left_half], attention_mask[:, -right_half:]], dim=-1)
        elif truncation == "error":
            raise NotImplementedError(f"{sequence_length=} is larger than {max_length=}")
        else:
            raise NotImplementedError(f"Unknown truncation method {truncation}")

    return input_ids, attention_mask


def tokenize_and_postprocess_data(
    messages: list[dict], tokenizer: PreTrainedTokenizer, max_length: int, pad_token_id: int, left_pad=True, truncation="error", hint_prompt: str = ""
):
    """Tokenize text and process outputs to consistent tensor shapes.

    Args:
        messages: List of dictionaries containing message content
        tokenizer: HuggingFace tokenizer instance
        max_length: Target sequence length
        pad_token_id: Padding token ID
        left_pad: Pad left if True
        truncation: Truncation strategy ("left"/"right"/"error")

    Returns:
        Tuple of (input_ids, attention_mask) from postprocess_data
    """
    if not (isinstance(messages, list) and len(messages) <= 2 and "content" in messages[0]):
        raise ValueError("messages must be a list containing at most 2 dictionaries with a 'content' key.")

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_with_hint = prompt + hint_prompt
    input_data = tokenizer(prompt_with_hint, return_tensors="pt", add_special_tokens=False)
    input_ids = input_data["input_ids"]
    attention_mask = input_data["attention_mask"]

    if len(hint_prompt) > 0:
        hint_data = tokenizer(hint_prompt, return_tensors="pt", add_special_tokens=False)
        hint_ids = hint_data["input_ids"]
        hint_length = hint_ids.shape[-1]
        hint_mask = torch.zeros_like(attention_mask)
        hint_mask[:, -hint_length:] = 1
    else:
        hint_length = 0
        hint_ids = torch.tensor([[]], dtype=torch.long)
        hint_mask = torch.zeros_like(input_ids)


    assert truncation in ["left", "right", "middle", "error"]
    assert input_ids.ndim == 2

    sequence_length = input_ids.shape[-1]
    if sequence_length < max_length:
        input_ids = pad_sequence_to_length(
            input_ids, max_seq_len=max_length, pad_token_id=pad_token_id, left_pad=left_pad
        )
        attention_mask = pad_sequence_to_length(
            attention_mask, max_seq_len=max_length, pad_token_id=0, left_pad=left_pad
        )
        hint_mask = pad_sequence_to_length(
            hint_mask, max_seq_len=max_length, pad_token_id=0, left_pad=left_pad
        )
    elif sequence_length > max_length:
        if truncation == "left":
            # actually, left truncation may not be reasonable
            input_ids = input_ids[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]
            hint_mask = hint_mask[:, -max_length:]
        elif truncation == "right":
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
            hint_mask = hint_mask[:, :max_length]
        elif truncation == "middle":
            left_half = max_length // 2
            right_half = max_length - left_half
            input_ids = torch.cat([input_ids[:, :left_half], input_ids[:, -right_half:]], dim=-1)
            attention_mask = torch.cat([attention_mask[:, :left_half], attention_mask[:, -right_half:]], dim=-1)
            hint_mask = torch.cat([hint_mask[:, :left_half], hint_mask[:, -right_half:]], dim=-1)
        elif truncation == "error":
            raise NotImplementedError(f"{sequence_length=} is larger than {max_length=}")
        else:
            raise NotImplementedError(f"Unknown truncation method {truncation}")

    return input_ids, attention_mask, hint_ids, hint_mask, prompt_with_hint


def tokenize_and_postprocess_data_deprecated(
    messages: list[dict], tokenizer: PreTrainedTokenizer, max_length: int, pad_token_id: int, left_pad=True, truncation="error", hint_prompt: str = ""
):
    """Tokenize text and process outputs to consistent tensor shapes.

    Args:
        messages: List of dictionaries containing message content
        tokenizer: HuggingFace tokenizer instance
        max_length: Target sequence length
        pad_token_id: Padding token ID
        left_pad: Pad left if True
        truncation: Truncation strategy ("left"/"right"/"error")

    Returns:
        Tuple of (input_ids, attention_mask) from postprocess_data
    """
    if not (isinstance(messages, list) and len(messages) <= 2 and "content" in messages[0]):
        raise ValueError("messages must be a list containing at most 2 dictionaries with a 'content' key.")

    input_ids_without_hint = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

    messages_with_hint = messages.copy()
    messages_with_hint[-1]["content"] = messages[-1]["content"] + hint_prompt

    prompt = tokenizer.apply_chat_template(messages_with_hint, add_generation_prompt=True, tokenize=False)
    input_data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = input_data["input_ids"]
    attention_mask = input_data["attention_mask"]

    hint_length = 0
    hint_ids = torch.tensor([[]], dtype=torch.long)
    hint_mask = torch.zeros_like(input_ids)

    if len(hint_prompt) > 0:
        len_with_hint = input_ids.shape[-1]
        len_without_hint = input_ids_without_hint.shape[-1]

        prefix_len = 0
        while (prefix_len < len_without_hint and 
               input_ids[0, prefix_len] == input_ids_without_hint[0, prefix_len]):
            prefix_len += 1

        suffix_len = 0
        while (suffix_len < len_without_hint and 
                input_ids[0, len_with_hint - 1 - suffix_len] == input_ids_without_hint[0, len_without_hint - 1 - suffix_len]):
            suffix_len += 1

        if prefix_len + suffix_len >= len_with_hint:
            hint_start_index = prefix_len
            hint_end_index = prefix_len
        else:
            hint_start_index = prefix_len
            hint_end_index = len_with_hint - suffix_len

        hint_mask[:, hint_start_index:hint_end_index] = 1
        hint_ids = input_ids[:, hint_start_index:hint_end_index]

    assert truncation in ["left", "right", "middle", "error"]
    assert input_ids.ndim == 2

    sequence_length = input_ids.shape[-1]
    if sequence_length < max_length:
        input_ids = pad_sequence_to_length(
            input_ids, max_seq_len=max_length, pad_token_id=pad_token_id, left_pad=left_pad
        )
        attention_mask = pad_sequence_to_length(
            attention_mask, max_seq_len=max_length, pad_token_id=0, left_pad=left_pad
        )
        hint_mask = pad_sequence_to_length(
            hint_mask, max_seq_len=max_length, pad_token_id=0, left_pad=left_pad
        )
    elif sequence_length > max_length:
        if truncation == "left":
            # actually, left truncation may not be reasonable
            input_ids = input_ids[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]
            hint_mask = hint_mask[:, -max_length:]
        elif truncation == "right":
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
            hint_mask = hint_mask[:, :max_length]
        elif truncation == "middle":
            left_half = max_length // 2
            right_half = max_length - left_half
            input_ids = torch.cat([input_ids[:, :left_half], input_ids[:, -right_half:]], dim=-1)
            attention_mask = torch.cat([attention_mask[:, :left_half], attention_mask[:, -right_half:]], dim=-1)
            hint_mask = torch.cat([hint_mask[:, :left_half], hint_mask[:, -right_half:]], dim=-1)
        elif truncation == "error":
            raise NotImplementedError(f"{sequence_length=} is larger than {max_length=}")
        else:
            raise NotImplementedError(f"Unknown truncation method {truncation}")

    return input_ids, attention_mask, hint_ids, hint_mask, prompt, messages_with_hint