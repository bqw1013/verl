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
import asyncio
import threading
import pandas as pd

from tqdm import tqdm
from datasets import Dataset
from openai import OpenAI, AsyncOpenAI
from transformers import AutoTokenizer
from typing import Union, List, Dict, Any, Optional

class AsyncRemoteServiceRollout:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "none",
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def __del__(self):
        if hasattr(self, "_loop") and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if hasattr(self, '_thread') and self._thread.is_alive():
            self._thread.join()
    
    def _create_async_client(self):
        return AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    async def _request_api_async(
        self,
        client: AsyncOpenAI,
        prompt: List[int],
        max_tokens: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs
    ):
        # if not isinstance(prompt, List):
        #     raise ValueError(f"prompt must be a list of integers, got {type(prompt)}")
        
        request_params = kwargs.copy()
        request_params.update({
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        })

        try:
            response = await client.completions.create(
                model=self.model_name,
                prompt=prompt,
                logprobs=True,
                extra_body={"return_tokens_as_token_ids": True},
                **request_params
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

    async def generate_async(
        self,
        prompts: Union[List[List[int]], torch.Tensor],
        max_tokens: List[int],
        pad_token_id: int = None,
        concurrency_limit: int = 32,
        pbar: Optional[tqdm] = None,
        **kwargs) -> Dict:

        processed_prompts = []
        if isinstance(prompts, torch.Tensor):
            if pad_token_id is None:
                raise ValueError("pad_token_id is required when prompts is a tensor")
            for prompt_tensor in prompts:
                mask = prompt_tensor != pad_token_id
                processed_prompts.append(prompt_tensor[mask].tolist())
        else:
            processed_prompts = prompts

        if len(processed_prompts) != len(max_tokens):
            raise ValueError("length of prompts and max_tokens must be the same")
        
        semaphore = asyncio.Semaphore(concurrency_limit)
        tasks = []

        async with self._create_async_client() as client:

            async def limited_request(prompt_data, mt_data):
                async with semaphore:
                    params = kwargs.copy()
                    params["max_tokens"] = mt_data
                    return await self._request_api_async(client, prompt_data, **params)

            for p, mt in zip(processed_prompts, max_tokens):
                task = asyncio.create_task(limited_request(p, mt))

                if pbar:
                    task.add_done_callback(lambda f: pbar.update(1))

                tasks.append(task)

            results = await asyncio.gather(*tasks)

        return results

    def generate(
        self,
        prompts: Union[List[List[int]], torch.Tensor],
        max_tokens: List[int],
        pad_token_id: int = None,
        concurrency_limit: int = 32,
        **kwargs) -> List[Dict]:
        num_prompts = len(prompts)

        with tqdm(total=num_prompts, desc="Generating") as pbar:
            future = asyncio.run_coroutine_threadsafe(
                self.generate_async(
                    prompts=prompts,
                    max_tokens=max_tokens,
                    pad_token_id=pad_token_id,
                    concurrency_limit=concurrency_limit,
                    pbar=pbar,
                    **kwargs
                ),
                self._loop
            )

            result = future.result()
            return pd.DataFrame(result).to_dict(orient='list')


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
        pad_token_id: int = None,
        concurrency_limit: int = 32,
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

        dataset = dataset.map(process_prompt, num_proc=concurrency_limit)
        return dataset.to_dict()

def test_service_rollout():
    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-fs/models/Qwen/Qwen2.5-7B-Instruct")
    prompt = "Hello, how are you?"
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
    prompt1 = prompt + tokenizer.encode("Fine! Thanks for asking.")
    prompt2 = prompt + tokenizer.encode("What's your name?")
    prompts = [prompt1, prompt2] * 500
    service_rollout = RemoteServiceRollout(
        base_url="https://u66551-baab-24341aca.cqa1.seetacloud.com:8443/v1",
        model_name="Qwen2.5-7B-Instruct",
    )
    rollouts = service_rollout.generate(prompts, num_proc=64, temperature=0, max_tokens=[2048] * len(prompts), top_p=1.0)
    pass

def test_async_service_rollout():
    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-fs/models/Qwen/Qwen2.5-7B-Instruct")
    prompt = "Hello, how are you?"
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
    prompt1 = prompt + tokenizer.encode("Fine! Thanks for asking.")
    prompt2 = prompt + tokenizer.encode("What's your name?")
    prompts = [prompt1, prompt2] * 500
    service_rollout = AsyncRemoteServiceRollout(
        base_url="https://u66551-baab-24341aca.cqa1.seetacloud.com:8443/v1",
        model_name="Qwen2.5-7B-Instruct",
    )
    rollouts = service_rollout.generate(prompts, max_tokens=[2048] * len(prompts), temperature=0, top_p=1.0, concurrency_limit=512)
    pass

tokenizer = AutoTokenizer.from_pretrained("/root/autodl-fs/models/Qwen/Qwen2.5-7B-Instruct")

def test():
    prompts = [151644,   8948,    198,   2610,    525,    264,  10950,  17847,
            13, 151645,    198, 151644,    872,    198,   5501,   2874,   3019,
           553,   3019,     11,    323,   2182,    697,   1590,   4226,   2878,
          1124,  79075,   6257,    624,     32,  21495,    304,    264,  80715,
         16184,  11031,    702,  17228,    320,     20,     11,    481,     17,
           701,    320,     16,     15,     11,    220,     20,      8,    323,
           320,     20,     11,    220,     20,    568,   2585,   1657,   9334,
          8153,    525,    304,    279,   3082,    315,    279,  21495,     30,
         17399,    697,   4226,    438,    264,  12122,    311,    279,  23480,
         55666,     13, 151645,    198, 151644,  77091,    198,   5501,   2874,
          3019,    553,   3019,     11,    323,   2182,    697,   1590,   4226,
          2878,   1124,  79075,   6257,    624,     32,  21495,    702,    264,
          2331,    315,    220,     17,     19,   9961,    323,    264,   2608,
           315,    220,     16,     23,   9961,     13,   1416,    220,     21,
         19766,     11,  63828,   3232,    307,  11202,  10747,    323,  18308,
         63828,   1948,    279,   1378,  23092,     11,    525,  67765,   1526,
           279,   1909,    315,    279,  21495,     11,   1128,    374,    279,
          2790,   3084,    315,    279,  19766,     30,   7036,     25,    279,
          2374,    315,    279,  19766,   1969,    387,   2686,   1091,    279,
          2331,    315,    279,  21495,    304,   1973,    311,   4946,   1526,
           279,  21495]
    service_rollout = AsyncRemoteServiceRollout(
        base_url="https://u66551-baab-24341aca.cqa1.seetacloud.com:8443/v1",
        model_name="Qwen2.5-7B-Instruct",
    )
    x = service_rollout.generate([prompts], max_tokens=[2048], temperature=1.0, top_p=0.8, concurrency_limit=512)
    pass

if __name__ == "__main__":
    # import time
    # start_time = time.time()
    # test_async_service_rollout()
    # end_time = time.time()
    # print(f"Time taken: {end_time - start_time} seconds")
    test()
    pass