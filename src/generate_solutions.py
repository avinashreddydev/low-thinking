from vllm import LLM, SamplingParams
import argparse
from datasets import load_dataset
from typing import Optional
import json
from tqdm import tqdm
import os
from thinking_extractor import extract_thinking
from transformers import AutoTokenizer
from openai import OpenAI
from utils import RESPONSE_EXTRACTOR, RESPONSE_COMPARATOR
from tokenizers import Tokenizer

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def model_and_tokenizer(
    model_name: str, return_tokenizer_only: bool = False
) -> tuple[Optional[LLM], Optional[AutoTokenizer]]:
    if return_tokenizer_only:
        return None, AutoTokenizer.from_pretrained(model_name)
    else:
        llm = LLM(model=model_name, gpu_memory_utilization=0.5, dtype="bfloat16")
        return llm, llm.get_tokenizer()


def offline_inference(llm: LLM, tokenizer: Tokenizer, prompt: str) -> str:

    thinking_end_token = "</think>"
    thinking_end_token = tokenizer.encode(thinking_end_token, add_special_tokens=False)[
        0
    ]

    sampling_params = SamplingParams(
        max_tokens=8192,
        temperature=0.7,
        top_p=0.95,
        top_k=100,
    )
    responses = llm.generate(prompt, sampling_params=sampling_params, use_tqdm=False)
    text = responses[0].outputs[0].text

    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    if thinking_end_token in text_tokens:
        thinking_response = text[: text_tokens.index(thinking_end_token)]
        answer_response = text[text_tokens.index(thinking_end_token) + 1 :]
    else:
        thinking_response = ""
        answer_response = text

    thinking_tokens_len = len(
        tokenizer.encode(thinking_response, add_special_tokens=False)
    )

    return thinking_response, answer_response, thinking_tokens_len


def summarize_thinking(
    llm: LLM, tokenizer: Tokenizer, problem: str, thinking: str
) -> str:
    sampling_params = SamplingParams(
        max_tokens=8192,
        temperature=0.7,
        top_p=0.95,
        top_k=100,
    )
    conv = [
        {
            "role": "system",
            "content": "You are a helpful assistant that summarizes thinking.",
        },
        {
            "role": "user",
            "content": f"Problem: {problem}\nThinking: {thinking} \n Please summarize the thinking in a few words. Directly output the summary, do not include any other text.",
        },
    ]

    templated_prompt = tokenizer.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=True
    )

    response = llm.generate(
        templated_prompt, sampling_params=sampling_params, use_tqdm=False
    )
    return response[0].outputs[0].text


def online_inference(
    client: OpenAI, model: str, prompt: str, tokenizer: Tokenizer
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=build_conv(prompt),
        stream=False,
    )

    thinking_response = response.choices[0].message.reasoning_content
    answer_response = response.choices[0].message.content
    thinking_tokens_len = len(
        tokenizer.encode(thinking_response, add_special_tokens=False)
    )
    return thinking_response, answer_response, thinking_tokens_len


def templated_prompt(problem: str, tokenizer: Tokenizer) -> str:
    conv = [
        {
            "role": "system",
            "content": "You are a math expert. You are given a math problem and you need to solve it.",
        },
        {
            "role": "user",
            "content": problem
            + "\n Please reason step by step, and put your final answer within \\boxed{}.",
        },
    ]

    templated_prompt = tokenizer.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=True
    )
    return templated_prompt


# def check_summary_thinking(problem: str, thinking: str) -> str:
#     bos = "<｜begin▁of▁sentence｜>"
#     sot = "<｜Assistant｜><think>\n"
#     eot = "\n</think>"
#     return bos + sot + thinking + eot


def check_summary_thinking(
    problem: str, thinking: str, llm: LLM, tokenizer: Tokenizer
) -> str:
    sampling_params = SamplingParams(
        max_tokens=8192,
        temperature=0.7,
        top_p=0.95,
        top_k=100,
    )
    conv = [
        {
            "role": "system",
            "content": "You are a math expert. You are given a math problem and you need to solve it.",
        },
        {
            "role": "user",
            "content": problem
            + "\n Please reason step by step, and put your final answer within \\boxed{}. Answer the question with a brief thinking and solve the problem.",
        },
    ]
    templated_prompt = tokenizer.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=True
    )
    templated_prompt += thinking + "\n</think>"

    response = llm.generate(
        templated_prompt, sampling_params=sampling_params, use_tqdm=False
    )

    return response[0].outputs[0].text


def construct_sft_text(
    prompt: str, thinking: str, answer: str, tokenizer: Tokenizer
) -> str:

    conv = [
        {
            "role": "system",
            "content": "You are a math expert. You are given a math problem and you need to solve it.",
        },
        {
            "role": "user",
            "content": prompt
            + "\n Please reason step by step, and put your final answer within \\boxed{}. Answer the question with a brief thinking and solve the problem.",
        },
        {
            "role": "assistant",
            "content": f"<think>\n{thinking}\n</think>{answer}",
        },
    ]

    sft_text = tokenizer.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=False
    )
    return sft_text


def online_inference(model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=build_conv(prompt),
        stream=False,
    )
    return response.choices[0].message.content


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reasoning_model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    )
    parser.add_argument(
        "--summarization_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct"
    )
    parser.add_argument(
        "--dataset", type=str, default="SynthLabsAI/Big-Math-RL-Verified"
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="data/generated")
    parser.add_argument("--output_filename", type=str, default="sft_data.jsonl")
    parser.add_argument("--eval_mode", type=bool, default=False)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--offline", type=bool, default=True)
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
            + "\n Please reason step by step, and put your final answer within \\boxed{}.",
        },
    ]
    return conv


def main():
    args = parse_args()

    answer_extractor = RESPONSE_EXTRACTOR[args.dataset]
    answer_comparator = RESPONSE_COMPARATOR[args.dataset]

    if args.offline:
        print("Offline mode")
        r_model, r_tokenizer = model_and_tokenizer(
            args.reasoning_model, return_tokenizer_only=False
        )
        s_model, s_tokenizer = model_and_tokenizer(
            args.summarization_model, return_tokenizer_only=False
        )
    else:
        print("Online mode")
        r_model, r_tokenizer = model_and_tokenizer(
            args.reasoning_model, return_tokenizer_only=False
        )
        s_model, s_tokenizer = model_and_tokenizer(
            args.summarization_model, return_tokenizer_only=False
        )
    dataset = load_dataset(args.dataset, split=args.split)
    #  Clear the output file
    open(args.output_dir + "/" + args.output_filename, "w").close()
    accuracy = 0
    samples_count = 0
    i = 660

    max_samples = len(dataset) if args.eval_mode else args.num_samples

    # introduce the tqdm to show the progress in while loop
    pbar = tqdm(total=max_samples, desc="Generating solutions")

    while samples_count < max_samples:
        problem = dataset[i]["problem"]
        answer = dataset[i]["answer"]

        if args.offline:
            thinking_response, answer_response, thinking_tokens_len = offline_inference(
                r_model, r_tokenizer, problem
            )
        else:
            thinking_response, answer_response, thinking_tokens_len = online_inference(
                client, args.reasoning_model, problem, s_tokenizer
            )

        predicted_answer = answer_extractor(answer_response)
        is_correct = answer_comparator(predicted_answer, answer)
        accuracy += is_correct

        data = {
            "id": i,
            "problem": problem,
            "answer": answer,
            "generated_answer": answer_response,
            "thinking": thinking_response,
            "thinking_tokens_len": thinking_tokens_len,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
        }

        if is_correct and thinking_tokens_len > 100 and not args.eval_mode:
            summary_thinking = summarize_thinking(
                s_model, s_tokenizer, problem, thinking_response
            )
            answer_with_summary_thinking = check_summary_thinking(
                problem, summary_thinking, r_model, r_tokenizer
            )

            new_answer = answer_extractor(answer_with_summary_thinking)
            is_new_correct = answer_comparator(new_answer, answer)

            if is_new_correct:
                data["summary_thinking"] = summary_thinking
                data["summary_thinking_tokens_len"] = len(
                    r_tokenizer.encode(summary_thinking, add_special_tokens=False)
                )
                data["answer_with_summary_thinking"] = answer_with_summary_thinking
                data["new_answer"] = new_answer
                data["new_is_correct"] = is_new_correct
                data["text"] = construct_sft_text(
                    problem, summary_thinking, answer_with_summary_thinking, s_tokenizer
                )

                with open(args.output_dir + "/" + args.output_filename, "a") as f:
                    f.write(json.dumps(data) + "\n")
                samples_count += 1
                pbar.update(1)
        else:
            if args.eval_mode:
                with open(args.output_dir + "/" + args.output_filename, "a") as f:
                    f.write(json.dumps(data) + "\n")
                samples_count += 1
                pbar.update(1)

        i += 1

    pbar.close()


if __name__ == "__main__":
    main()
