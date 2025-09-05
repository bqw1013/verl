# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the for the specific language governing permissions and
# limitations under the License.
import torch

from openai import OpenAI
from datasets import Dataset
from transformers import AutoTokenizer
from typing import Union, List, Dict, Any


class RemoteServiceRollout:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "none",
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key

    def _create_client(self):
        return OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _request_api(
        self, 
        prompt: List[int], 
        max_tokens: int = 4096, 
        temperature: float = 1.0, 
        top_p: float = 0.9, 
        **kwargs
    ):
        if not isinstance(prompt, List):
            raise ValueError(f"prompt must be a list of integers, got {type(prompt)}")
        
        kwargs.update({
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        })

        client = self._create_client()

        try:
            response = client.completions.create(
                model=self.model_name,
                prompt=prompt,
                logprobs=True,
                extra_body={"return_tokens_as_token_ids": True},
                **kwargs
            )
            return {
                "status": "success",
                "text": response.choices[0].text,
                "logprobs": response.choices[0].logprobs.token_logprobs,
                "token_ids": [int(item.split(":")[1]) for item in response.choices[0].logprobs.tokens],
                "error": ""
            }
        except Exception as e:
            return {
                "status": "failure",
                "text": "",
                "logprobs": [],
                "token_ids": [],
                "error": str(e)
            }
        finally:
            client.close()

    def generate(
        self, 
        prompts: Union[List[List[int]], torch.Tensor],
        max_tokens: List[int],
        num_proc: int = 8,
        pad_token_id: int = None,
        **kwargs) -> List[Dict]:
        if isinstance(prompts, torch.Tensor):
            if pad_token_id is None:
                raise ValueError("pad_token_id is required when prompts is a tensor")
            for prompt in prompts:
                mask = prompt != pad_token_id
                prompt = prompt[mask]
                prompt = prompt.tolist()
            dataset = Dataset.from_dict({"prompt": prompts, "max_tokens": max_tokens})
        else:
            dataset = Dataset.from_dict({"prompt": prompts, "max_tokens": max_tokens})

        def process_prompt(item):
            params = kwargs.copy()
            params["max_tokens"] = item["max_tokens"]
            return self._request_api(item["prompt"], **params)

        dataset = dataset.map(process_prompt, num_proc=num_proc)
        return dataset.to_dict()

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-fs/models/Qwen/Qwen2.5-7B-Instruct")
    prompt = "Hello, how are you?"
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
    prompt1 = prompt + tokenizer.encode("Fine! Thanks for asking.")
    prompt2 = prompt + tokenizer.encode("What's your name?")
    prompts = [prompt1, prompt2] * 5
    service_rollout = RemoteServiceRollout(
        base_url="https://u66551-baab-24341aca.cqa1.seetacloud.com:8443/v1",
        model_name="Qwen2.5-7B-Instruct",
    )
    rollouts = service_rollout.generate(prompts, num_proc=1, temperature=0, max_tokens=[4096] * len(prompts), top_p=1.0)
    pass
    # response = rollout._request_api(prompt, temperature=1.0, max_tokens=4096, top_p=0.9)