import json
import argparse
from scoring_functions import countdown
from utils import load_dataset, generate_responses

def calc_countdown_accuracy(responses: list[str], ground_truths: list[str]) -> float:
    if isinstance(responses[0], list):
        responses = [response[0] for response in responses]
    scores = [countdown.compute_score(response, ground_truth, format_score=0.0) for response, ground_truth in zip(responses, ground_truths)]
    return sum(scores) / len(scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset_path", type=str, default="/root/autodl-tmp/code/verl/datasets/countdown/test.parquet")
    parser.add_argument("--prompt_key", type=str, default="question")
    parser.add_argument("--ground_truth_key", type=str, default="reward_model")
    parser.add_argument("--template", type=str, default="qwen")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--output_file", type=str, default="countdown_results.jsonl")
    args = parser.parse_args()
    
    generation_config = {
        "n": args.n,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
    }
    prompts, ground_truths = load_dataset(args.dataset_path, prompt_key=args.prompt_key, ground_truth_key=args.ground_truth_key, template=args.template)
    ground_truths = [item["ground_truth"] for item in ground_truths]
    responses = generate_responses(prompts, args.model_path, "online", generation_config=generation_config)
    acc = calc_countdown_accuracy(responses, ground_truths)
    with open(args.output_file, "a") as f:
        f.write(json.dumps({"model": args.model_path, "accuracy": acc}) + "\n")

if __name__ == "__main__":
    main()