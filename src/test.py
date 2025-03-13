from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

messages = [
    {
        "role": "user",
        "content": "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$. Please reason step by step and put your final answer within \\boxed{}.",
    }
]
response = client.chat.completions.create(model=model, messages=messages, stream=False)

print("###### Reasoning ######")
print(response.choices[0].message.reasoning_content)

print("###### Answer ######")
print(response.choices[0].message.content)
