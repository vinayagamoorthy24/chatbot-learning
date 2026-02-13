import os
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")   # 🔐 hidden
)

document = [
    "india has income tax laws.",
    "tax is calculation based income tax slabs.",
    "gst is only 18 for many goods ",
]

questions = "EXPLAIN GST?"

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[
        {"role": "system", "content": "answer only from the given document"},
        {"role": "user", "content": f"Document: {document}\n\nQuestion: {questions}"}
    ],
)

print(response.choices[0].message.content)
