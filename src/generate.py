from vllm import LLM
from datasets import load_dataset

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--dataset", type=str, default="SynthLabsAI/Big-Math-RL-Verified"
    )
    parser.add_argument("--output_dir", type=str, default="data/generated")
    parser.add_argument(
        "--output_filename", type=str, default="bigmath_low_thinking.jsonl"
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main(args):
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
        tensor_parallel_size=1,
    )

    tokenizer = llm.get_tokenizer()

    dataset = load_dataset(args.dataset, split=args.split)

    for i in range(len(dataset)):
        problem = dataset[i]["problem"]
        solution = dataset[i]["solution"]


if __name__ == "__main__":

    args = parse_args()
