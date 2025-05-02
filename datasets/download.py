# -*- coding:utf-8 -*-
"""
Author: Qiangwei Bai
Date: 2025-04-12 12:53:04
LastEditTime: 2025-04-12 12:56:51
LastEditors: Qiangwei Bai
FilePath: /verlx/datasets/download.py
Description: 
"""
import os
import argparse
from datasets import load_dataset

# 定义可用的数据集和其对应的Hugging Face地址
DATASETS = {
    "GSM8K": "gsm8k",
    "MATH": "nlile/hendrycks-MATH-benchmark",
    "AMC": "AI-MO/aimo-validation-amc",
    "AIME2024": "Maxwell-Jia/AIME_2024",
    "AoPS": "sparsh35/aops",
    "OlympiadBench": "knoveleng/OlympiadBench",
    "MinervaMath": "math-ai/minervamath",
    "DAPO-Math": "BytedTsinghua-SIA/DAPO-Math-17k"
}

def download_dataset(dataset_name):
    """下载指定的数据集并保存到对应目录"""
    if dataset_name not in DATASETS:
        print(f"错误：未找到数据集 '{dataset_name}'。可用的数据集有：{', '.join(DATASETS.keys())}")
        return False
    
    dataset_id = DATASETS[dataset_name]
    print(f"正在下载 {dataset_name} 数据集...")
    
    # 创建数据集目录
    dataset_dir = os.path.join("math", dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    try:
        # 下载数据集，GSM8K使用main子集
        if dataset_name == "GSM8K":
            dataset = load_dataset(dataset_id, "main")
        else:
            dataset = load_dataset(dataset_id)
        
        # 保存数据集到本地
        dataset.save_to_disk(dataset_dir)
        print(f"{dataset_name} 数据集已成功下载并保存到 {dataset_dir}")
        return True
    except Exception as e:
        print(f"下载 {dataset_name} 数据集时出错: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="下载指定的数据集")
    parser.add_argument("datasets", nargs="*", help="要下载的数据集名称，可指定多个")
    parser.add_argument("--list", action="store_true", help="列出所有可用的数据集")
    args = parser.parse_args()
    
    # 列出可用数据集
    if args.list:
        print("可用的数据集：")
        for name in DATASETS:
            print(f"- {name}")
        return
    
    # 检查是否提供了数据集名称
    if not args.datasets:
        print("错误：请指定至少一个数据集名称或使用 --list 查看可用数据集")
        print("示例: python datasets/download.py GSM8K MATH")
        parser.print_help()
        return
    
    success_count = 0
    for dataset_name in args.datasets:
        if download_dataset(dataset_name):
            success_count += 1
    
    print(f"下载完成！成功下载 {success_count}/{len(args.datasets)} 个数据集。")

if __name__ == "__main__":
    main()
