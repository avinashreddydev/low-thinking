from transformers import AutoTokenizer
from tokenizers import Tokenizer
from vllm import LLM, SamplingParams


def templated_prompt(
    prompt: str, tokenizer: Tokenizer, add_generation_prompt: bool = True
) -> str:
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
        {
            "role": "assistant",
            "content": "Let's think step by step.",
        },
    ]

    templated_prompt = tokenizer.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    return templated_prompt


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )

    # llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", dtype="bfloat16")
    sampling_params = SamplingParams(
        max_tokens=8192,
        temperature=0.7,
        top_p=0.95,
        top_k=100,
    )

    prompt = "The proper divisors of 12 are 1, 2, 3, 4 and 6. A proper divisor of an integer $N$ is a positive divisor of $N$ that is less than $N$. What is the sum of the proper divisors of the sum of the proper divisors of 284?"
    # input_prompt = templated_prompt(prompt, tokenizer, add_generation_prompt=True)
    # print(input_prompt)
    # responses = llm.generate(input_prompt, sampling_params=sampling_params)
    # text = responses[0].outputs[0].text
    # print(text)
    # input_prompt = templated_prompt(prompt, tokenizer, add_generation_prompt=False)
    # input_prompt += "<｜Assistant｜>\n"
    # responses = llm.generate(input_prompt, sampling_params=sampling_params)
    # text = responses[0].outputs[0].text
    # print(text)

    input_prompt = templated_prompt(prompt, tokenizer, add_generation_prompt=False)
    print(input_prompt)
