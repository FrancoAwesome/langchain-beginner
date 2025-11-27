import subprocess
import json
import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("OLLAMA_BASE_URL")

# 调用 Ollama CLI 命令生成文本
command = [
    "ollama", "run", "deepseek-r1:1.5b"
]
result = subprocess.run(command, input="who are you?", capture_output=True, text=True)
print(result)
# 解析 JSON 响应
if result.returncode == 0:
    response = json.loads(result.stdout)
    print(response)
else:
    print("Error:", result.stderr)