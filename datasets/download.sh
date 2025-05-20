#!/bin/bash

set -e

echo "开始下载数据集"

# 执行Python下载脚本，指定GSM8K和MATH数据集
python download.py "AMC" "AIME2024" "OlympiadBench"
echo "下载脚本执行完成！"