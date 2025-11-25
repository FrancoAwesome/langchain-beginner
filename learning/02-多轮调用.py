import os
from dotenv import load_dotenv
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()
api_key = os.getenv('API_KEY_DASHSCOPE')
base_url = os.getenv('BASE_URL')
model_name = os.getenv('MODEL_NAME')

llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)

msg=[
    SystemMessage(content='你是一个个人助理，名字叫做琳达'),
    HumanMessage(content='我的名字是小沈'),
    AIMessage(content='不好意思，无法获取天气'),
    HumanMessage(content='天气怎么样，我是谁'),
]

response = llm.invoke(msg)
print(response)