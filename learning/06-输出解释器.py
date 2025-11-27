import os
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, XMLOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
api_key = os.getenv('API_KEY_DASHSCOPE')
base_url = os.getenv('BASE_URL')
model_name = os.getenv('MODEL_NAME')

llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)

prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个专业程序员'),
    ('user', '{use_data}')
])

output = JsonOutputParser()
# output = StrOutputParser()
# output = XMLOutputParser()

chain = prompt | llm | output

res = chain.invoke({'use_data': 'langchain是什么? 问题用question 回答用ans返回一个JSON格式'})
# res = chain.invoke({'use_data': 'langchain是什么?'})
print(res)