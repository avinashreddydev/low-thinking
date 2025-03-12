from vllm import LLM, SamplingParams
import argparse
from datasets import load_dataset
from typing import Optional
import json
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )
    parser.add_argument(
        "--dataset", type=str, default="SynthLabsAI/Big-Math-RL-Verified"
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="data/generated")
    parser.add_argument(
        "--output_filename", type=str, default="bigmath_rl_solutions.jsonl"
    )
    return parser.parse_args()


def build_conv(prompt: str) -> str:

    conv = [
        {
            "role": "system",
            "content": "You are a math expert. You are given a math problem and you need to solve it.",
        },
        {
            "role": "user",
            "content": prompt
            + "\n Please reason step by step, and put your final answer within \boxed{}.",
        },
    ]
    return conv


def main():
    args = parse_args()

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.9,
        top_k=50,
        max_tokens=8192,
    )

    tokenizer = llm.get_tokenizer()

    dataset = load_dataset(args.dataset, split=args.split)

    #  Clear the output file
    open(args.output_dir + "/" + args.output_filename, "w").close()

    for i in tqdm(range(len(dataset))):
        problem = dataset[i]["problem"]
        answer = dataset[i]["answer"]

        tokenized_prompt = tokenizer.apply_chat_template(
            build_conv(problem), tokenize=False, add_generation_prompt=True
        )
        print(tokenized_prompt)

        responses = llm.generate(tokenized_prompt, sampling_params, use_tqdm=False)

        final_response = responses[0].outputs[0].text

        data = {
            "id": i,
            "problem": problem,
            "answer": answer,
            "generated_answer": final_response,
            "tokenized_prompt": tokenized_prompt,
        }

        with open(args.output_dir + "/" + args.output_filename, "a") as f:
            f.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    main()
