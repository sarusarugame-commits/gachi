import os
from openai import OpenAI

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("No GROQ_API_KEY found")
    exit()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key,
)

try:
    models = client.models.list()
    print("Available Models:")
    for m in models:
        print(m.id)
except Exception as e:
    print(f"Error: {e}")
