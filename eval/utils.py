import torch
import pandas as pd
from openai import OpenAI
from datasets import Dataset
from transformers import AutoTokenizer

gpu_num = torch.cuda.device_count()

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

def load_dataset(dataset_path: str, prompt_key: str = "prompt", ground_truth_key: str = "answer", template: str = "qwen"):
    df = pd.read_parquet(dataset_path)

    if len(df) == 0:
        return [], []

    prompts = df[prompt_key].tolist()
    ground_truths = df[ground_truth_key].tolist()

    if len(prompts) != len(ground_truths):
        raise ValueError(f"Length of prompts and ground_truths must be the same, but got {len(prompts)} and {len(ground_truths)}")

    if isinstance(prompts[0], str):
        if template in ["qwen_math"]:
            prompts = [[
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": prompt}
            ] for prompt in prompts]
        elif template in ["qwen"]:
            prompts = [[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{}.\n" + prompt}
            ] for prompt in prompts]
        else:
            raise ValueError(f"Unsupported template: {template}")
    
    if isinstance(prompts[0][0], dict) and "content" in prompts[0][0]:
        return prompts, ground_truths
    else:
        raise ValueError(f"Unsupported prompts type: {type(prompts[0])}")


def get_completion(messages: list[dict], **kwargs):
    try:
        response = client.chat.completions.create(
            model="Qwen",
            messages=messages,
            **kwargs
        )
        return [choice.message.content.strip() for choice in response.choices]
    except Exception as e:
        print(f"Error: {e}")
        return [""] * kwargs["n"]


def generate_responses(prompts: list[list[dict]],
                      model_path: str,
                      inference_mode: str,
                      tokenizer: AutoTokenizer = None,
                      generation_config: dict = None):
    if inference_mode == "online":
        prompts = [{"messages": prompt} for prompt in prompts]
        dataset = Dataset.from_list(prompts)
        dataset = dataset.map(lambda x: {"response": get_completion(x["messages"], **generation_config)}, num_proc=gpu_num * 8)
        return dataset["response"]
    else:
        raise NotImplementedError("Not implemented")