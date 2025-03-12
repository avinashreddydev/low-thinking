from constants.prompts import THINKING_SUMMARY_PROMPT, MATH_PROMPT


class SummaryGenerator:
    def __init__(self, model, tokenizer, evaluator):
        self.model = model
        self.tokenizer = tokenizer

    def think_low(self, problem, solution):
        pass

    def templated_prompt(self, problem: str):
        conv = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": problem + MATH_PROMPT},
        ]
        tokenized_prompt = self.tokenizer.apply_chat_template(conv, tokenize=False)
        return tokenized_prompt

    def generate_summary(self, problem: str, solution: str):
        pass


if __name__ == "__main__":
    pass
