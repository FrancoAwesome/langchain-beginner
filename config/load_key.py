import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
api_key = os.getenv('API_KEY_DASHSCOPE')
print(api_key)
base_url = os.getenv('BASE_URL')
model_name = os.getenv('MODEL_NAME')

def load_envkey():
    return {
        'api_key': api_key,
        'base_url': base_url,
        'model_name': model_name
    }
