# -*- coding:utf-8 -*-
"""
Author: Qiangwei Bai
Date: 2025-04-22 06:16:03
LastEditTime: 2025-04-22 06:16:03
LastEditors: Qiangwei Bai
FilePath: /verlx/verl/utils/reward_score/simplerl.py
Description: 
"""
import re

from verl.utils.reward_score.qwen_math_eval_toolkit.parser import extract_answer as qwen_extract_answer
from verl.utils.reward_score.qwen_math_eval_toolkit.grader import math_equal as qwen_math_equal
# from .qwen_math_eval_toolkit.parser import extract_answer as qwen_extract_answer
# from .qwen_math_eval_toolkit.grader import math_equal as qwen_math_equal

def extract_last_boxed(text):
    """
    提取 LaTeX 文本中最后一个 \boxed 命令中的内容
    
    返回:
    - str: 最后一个 \boxed 中的内容。如果没有找到则返回 None
    """
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    
    # 找到所有匹配
    matches = list(re.finditer(pattern, text))
    
    # 如果找到匹配，返回最后一个的内容
    if matches:
        return matches[-1].group(0)
    return None

def extract_solution(solution_str):
    model_output= re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', solution_str, flags=re.DOTALL,count = 1)
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"] 
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    
    predict_answer = qwen_extract_answer(model_output, data_name="math")
    extract_boxed_answer = extract_last_boxed(model_output)
    # True means the boxed answer is correct
    if extract_boxed_answer is not None:
        return predict_answer, True
    else:
        return predict_answer, False

def compute_score(solution_str: str, ground_truth: str):
    extract_answer, is_boxed_matched = extract_solution(solution_str=solution_str)
    correct = qwen_math_equal(prediction=extract_answer, reference=ground_truth)
    if correct:
        box_match = 1.0
    else:
        box_match = -0.5
    if not is_boxed_matched:
        box_match =  -1.0
    return box_match

if __name__ == "__main__":
    solution_str = "To determine how many feet less than Martha Jim walks, we need to calculate the distance each of them walks and then find the difference.\n\nFirst, let's calculate the distance Martha walks. Martha walks along the length and width of the rectangular field. The length of the field is 400 feet and the width is 300 feet. Therefore, the total distance Martha walks is:\n\\[\n400 + 300 = 700 \\text{ feet}\n\\]\n\nNext, let's calculate the distance Jim walks. Jim walks diagonally across the rectangular field. The diagonal of a rectangle can be found using the Pythagorean theorem, which states that in a right-angled triangle, the square of the hypotenuse (the diagonal in this case) is equal to the sum of the squares of the other two sides. So, the diagonal \\(d\\) is given by:\n\\[\nd = \\sqrt{400^2 + 300^2}\n\\]\nCalculating the squares, we get:\n\\[\n400^2 = 160000 \\quad \\text{and} \\quad 300^2 = 90000\n\\]\nAdding these together, we get:\n\\[\n160000 + 90000 = 250000\n\\]\nTaking the square root of 250000, we get:\n\\[\nd = \\sqrt{250000} = 500 \\text{ feet}\n\\]\n\nNow, we need to find out how many feet less than Martha Jim walks. We do this by subtracting the distance Jim walks from the distance Martha walks:\n\\[\n700 - 500 = 200 \\text{ feet}\n\\]\n\nTherefore, Jim walks \\(\\boxed{200}\\) feet less than Martha."
    ground_truth = "200"
    print(compute_score(solution_str, ground_truth))
    pass
