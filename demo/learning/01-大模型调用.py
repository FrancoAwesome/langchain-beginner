import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
api_key = os.getenv('API_KEY_DASHSCOPE')
print(api_key)
base_url = os.getenv('BASE_URL')
model_name = os.getenv('MODEL_NAME')

llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)
response = llm.invoke('你好，请问你是谁')
print(response)

print('*'*30)
print(response.content)
print('*'*30)