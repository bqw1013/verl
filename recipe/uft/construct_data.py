# -*- coding:utf-8 -*-
import pandas as pd
import re
from verl.utils.reward_score.math_verify import compute_score

data = pd.read_parquet("data.parquet")
# data = data[data["response"].apply(lambda res: "</think>" in res)]
# data["response_with_think"] = data["response"]
# data["response"] = data["response"].apply(lambda res: re.search(r"<think>(.*)<\/think>", res, re.DOTALL).group(1).strip() if re.search(r"<think>(.*)<\/think>", res, re.DOTALL) else "")
# data = data[data["response"].apply(len)!=0]
# data = data[data.apply(lambda row: compute_score(row["response"], row["clean_answer"])==1.0, axis=1)]
# data.index = range(len(data))

def merge_short_paragraphs(paragraphs: list[str], min_length: int = 100, joiner: str = "") -> list[str]:
    if not paragraphs:
        return []

    merged_paragraphs = [paragraphs[0]]

    for i in range(1, len(paragraphs)):
        current_para = paragraphs[i]
        last_merged_para = merged_paragraphs[-1]

        if len(current_para.split()) < min_length or len(last_merged_para.split()) < min_length:
            merged_paragraphs[-1] = last_merged_para + joiner + current_para
        else:
            merged_paragraphs.append(current_para)

    return merged_paragraphs

def build_hint(response: str) -> list[str]:
    hints = []
    paras = response.split("\n\n")
    for para in paras:
        sents = para.split("\n")
        hints.extend([sent + "\n" for sent in sents])
        hints[-1] = hints[-1] + "\n"
    hints[-1] = hints[-1].strip()
    hints = merge_short_paragraphs(hints, min_length=30, joiner=" ")
    return hints

data["hint"] = data["response"].apply(lambda res: build_hint(res))
data = data[data["hint"].apply(lambda hint: max([len(h.split()) for h in hint])<1000)]

prefix = "Please reason step by step, and put your final answer within \\boxed{}.\n"
data["prompt"] = data["problem"].apply(lambda p: [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prefix + p}])
data["reward_model"] = data["clean_answer"].apply(lambda x: {"ground_truth": x})

test = data.sample(n=500, random_state=42)
train = data.drop(test.index)
train.to_parquet("datasets/math/miromind-m1-sft-rl-fusion/train.parquet", index=False)
test.to_parquet("datasets/math/miromind-m1-sft-rl-fusion/test.parquet", index=False)
pass