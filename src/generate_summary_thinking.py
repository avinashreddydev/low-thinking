from openai import OpenAI
import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/generated")
    return parser.parse_args()


def main():
    args = parse_args()
    data = []
    with open(args.output_dir + "/" + args.input_file, "r") as f:
        for line in f:
            data.append(json.loads(line))
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    pbar = tqdm(total=len(data), desc="Generating summary of thinking")

    thinking_tokens_len_list = []
    summary_tokens_len_list = []
    for item in data:
        problem = item["problem"]
        thinking = item["thinking"]
        thinking_tokens_len = len(tokenizer.encode(thinking, add_special_tokens=False))
        thinking_tokens_len_list.append(thinking_tokens_len)

        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"Problem: {problem}\nThinking: {thinking} \n\n Please summarize the given thinking. Directly output the summary, without any other text.",
                },
            ],
        )

        summary = response.choices[0].message.content
        summary_tokens_len = len(tokenizer.encode(summary, add_special_tokens=False))
        summary_tokens_len_list.append(summary_tokens_len)
        item["summary"] = summary
        item["summary_tokens_len"] = summary_tokens_len
        pbar.update(1)
    with open(args.output_dir + f"/{args.input_file[:-5]}_summary.jsonl", "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    plt.figure(figsize=(10, 5))
    # Bar plot for thinking tokens length mean and Summary tokens length mean
    plt.bar(
        ["Thinking", "Summary"],
        [np.mean(thinking_tokens_len_list), np.mean(summary_tokens_len_list)],
    )
    plt.savefig(
        args.output_dir + f"/{args.input_file[:-5]}_summary_length_bar_plot.png"
    )


if __name__ == "__main__":
    print("Make Sure the model is set to non-thinking model")
    main()
