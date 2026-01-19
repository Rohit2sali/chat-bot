import os
from groq import Groq

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY not found in environment")

client = Groq(api_key=api_key)

def chat_with_llm(user_input: str, context: list[str]):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    for msg in context:
        messages.append({"role": "user", "content": msg})

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )

    return response.choices[0].message.content

