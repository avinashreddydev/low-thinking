import os


os.environ["WANDB_PROJECT"] = (
    "grpo_qwen_math_500_hard_len_512_epoch_1"  # name your W&B project
)
# os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import re
import torch
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl.trainer import GRPOConfig, GRPOTrainer
import wandb
from utils import DATASET_KEYS, RESPONSE_COMPARATOR, RESPONSE_EXTRACTOR
from transformers import AutoTokenizer
import pdb
import sys
from parser_qwen import extract_answer, strip_string

R1_STYLE_SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <reasoning> </reasoning> and
<answer> </answer> tags, respectively, i.e., <reasoning> reasoning process here </reasoning>
<answer>answer</answer>."""

TASK_SPECIFIC_INSTRUCTIONS = """Regardless of the approach, always conclude with: $answer$. Where [answer] is just the final number or expression that solves the problem."""


def preprocess_dataset(dataset_name, split="test", chunk_size=500) -> Dataset:
    dataset = load_dataset(dataset_name)[split]
    dataset = dataset.filter(lambda example: example["level"] in [5])

    def extract_hash_answer(text: str) -> str | None:
        try:
            return text.split("####")[1].strip()
        except IndexError:
            return None

    def process_batch(batch):

        prompts = [
            [
                {"role": "system", "content": R1_STYLE_SYSTEM_PROMPT},
                {"role": "user", "content": "Compute $99^2+99+1$ in your head."},
                {
                    "role": "assistant",
                    "content": "<reasoning>To solve this problem, we factor the first two terms, we have: $99^2+99+1=99(99+1)+1=99*100+1=9900+1=9901$.</reasoning>\n<answer>9901</answer>",
                },
                {"role": "user", "content": q.strip()},
            ]
            for q in batch["problem"]
        ]

        return {"prompt": prompts, "answer": batch["answer"]}

    return dataset.map(process_batch, batched=True, batch_size=chunk_size)


def extract_reasoning_tokens(text: str) -> str:
    """Extracts the text between <reasoning> and </reasoning> tags."""
    try:
        reasoning = text.split("<reasoning>")[-1].split("</reasoning>")[0].strip()
        return reasoning
    except IndexError:
        return ""


def extract_xml_answer(text: str) -> str:
    try:
        answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
        return strip_string(answer)
    except IndexError:
        return ""


def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has the correct format."""
    pattern = r"^<reasoning>(?:(?!</reasoning>).)*</reasoning>\n<answer>(?:(?!</answer>).)*</answer>$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(re.match(pattern, r, re.DOTALL)) for r in responses]
    # pdb.set_trace()
    return [1.0 if match else 0.0 for match in matches]


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward function that checks if the answer is correct."""
    responses = [completion[0]["content"] for completion in completions]
    # print(responses)
    extracted_responses = [strip_string(extract_xml_answer(r)) for r in responses]
    reasoning_tokens = [extract_reasoning_tokens(r) for r in responses]
    reasoning_lengths = [len(rt.split()) for rt in reasoning_tokens if rt]
    # pdb.set_trace()
    # Log the reasoning tokens to wandb for tracking (you can aggregate or limit this logging as needed)
    if reasoning_lengths:
        avg_reasoning_length = sum(reasoning_lengths) / len(reasoning_lengths)
        max_reasoning_length = max(reasoning_lengths)
        min_reasoning_length = min(reasoning_lengths)
    else:
        avg_reasoning_length = 0
        max_reasoning_length = 0
        min_reasoning_length = 0

    # Log these summary statistics to wandb
    wandb.log(
        {
            "avg_reasoning_tokens_length": avg_reasoning_length,
            "max_reasoning_tokens_length": max_reasoning_length,
            "min_reasoning_tokens_length": min_reasoning_length,
        }
    )
    # pdb.set_trace()
    answer = [strip_string(a) for a in answer]

    print(
        f"\n\n===============================================================\n"
        # f"User Question:\n{prompts[0][-1]['content']}"
        f"\n\nCorrect Answer:\n{answer[0]}\n"
        f"\n---------------------------------------------------------------\n"
        # f"\n\n1st/{len(completions)} generated responses:\n{responses[0]}"
        f"\n\nExtracted: {extracted_responses[0]}"
        f"\n\nCorrectness of all {len(completions)} responses: "
        + "".join(
            "Y" if RESPONSE_COMPARATOR["di-zhang-fdu/MATH500"](r, a) == True else "N"
            for r, a in zip(extracted_responses, answer)
        )
    )
    # pdb.set_trace()
    reward_list = [
        2.0 if RESPONSE_COMPARATOR["di-zhang-fdu/MATH500"](r, a) == True else 0.0
        for r, a in zip(extracted_responses, answer)
    ]
    # pdb.set_trace()
    return reward_list


def main():
    dataset_name = "di-zhang-fdu/MATH500"
    dataset = preprocess_dataset(dataset_name)
    # dataset_train = dataset.shuffle(seed=42).select(range(400))
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    dataset_train = split_dataset["train"]
    dataset_test = split_dataset["test"]

    print(f"Length of train dataset = {len(dataset_train['prompt'])}")
    # pdb.set_trace()

    model_name = "Qwen2.5-3B-Instruct"

    output_dir = "./output_hard_len_2048_epoch_1"
    run_name = f"{model_name}-{dataset_name.split('/')[-1]}"

    # Set memory-related environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    training_args = GRPOConfig(
        learning_rate=1e-5,
        beta=0.005,  # divergence coefficient – how much the policy is allowed to deviate from the reference model. higher value – more conservative updates. Default is 0.04
        optim="adamw_8bit",
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=4,
        num_generations=4,  # group size
        gradient_accumulation_steps=4,
        # max_prompt_length = 256,
        max_completion_length=2048,
        num_train_epochs=3,
        save_steps=100,
        max_grad_norm=0.5,
        report_to="wandb",
        output_dir=output_dir,
        run_name=run_name,
        log_on_each_node=False,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        torch_dtype=torch.bfloat16,
        # attn_implementation = "flash_attention_2",
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        model_max_length=training_args.max_completion_length,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward_func, correctness_reward_func],
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
    )

    # wandb.init(project="deepseek_r1_zero_grpo", name=run_name, mode="offline")  # name specifies job name
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
