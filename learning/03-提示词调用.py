import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPrompt, ChatPromptTemplate
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()
api_key = os.getenv('API_KEY_DASHSCOPE')
base_url = os.getenv('BASE_URL')
model_name = os.getenv('MODEL_NAME')

llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)

prompt = ChatPromptTemplate.from_messages([
    ('system','你是一个专业的技术文档编写者'),
    ('user', '请写一篇关于{use_data}的文档')
])

# chain 调用语句
chain = prompt | llm

response = chain.invoke({'user_data': 'java'})
print(response)