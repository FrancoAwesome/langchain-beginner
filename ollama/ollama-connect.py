import requests
import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("OLLAMA_BASE_URL")
model = "qwen2.5:7b"
payload = {
    "model": model,
    "prompt": "你好！",
    "max_tokens": 100,
    "stream": False
}

response = requests.post(url, json=payload)
print(response.json())