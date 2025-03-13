from tokenizers import Tokenizer
from transformers import AutoTokenizer
from typing import Tuple
import torch.distributed as dist
from vllm import LLM


def extract_thinking(
    response: str, thinking_stop_token: str, tokenizer: Tokenizer
) -> Tuple[str, str]:
    response_token_ids = tokenizer.encode(response, add_special_tokens=False)
    thinking_token_id = tokenizer.encode(thinking_stop_token, add_special_tokens=False)[
        -1
    ]
    if thinking_token_id not in response_token_ids:
        return {"text": "", "token_ids": []}, {
            "text": response,
            "token_ids": response_token_ids,
        }
    else:
        thinking_start_idx = response_token_ids.index(thinking_token_id)
        thinking_token_ids = response_token_ids[:thinking_start_idx]
        thinking = tokenizer.decode(thinking_token_ids)
        answer_token_ids = response_token_ids[thinking_start_idx + 1 :]
        answer = tokenizer.decode(answer_token_ids)

        thinking_data = {"text": thinking, "token_ids": thinking_token_ids}
        answer_data = {"text": answer, "token_ids": answer_token_ids}

        return thinking_data, answer_data


if __name__ == "__main__":

    # tokenizer = AutoTokenizer.from_pretrained(
    #     "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # )
    llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", enforce_eager=True)
    tokenizer = llm.get_tokenizer()
    thinking_stop_token = "</think>"

    sample_response = "So i need to find the answer to the question. Let's think step by step and then give the answer. </think> The answer is \\boxed{42}."
    thinking_data, answer_data = extract_thinking(
        sample_response, thinking_stop_token, tokenizer
    )
    print(thinking_data)
    print(answer_data)

    if dist.is_initialized():
        dist.destroy_process_group()
