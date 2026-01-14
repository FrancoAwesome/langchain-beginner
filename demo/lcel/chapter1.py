import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
api_key = os.getenv('API_KEY_DASHSCOPE')
base_url = os.getenv('BASE_URL')
model_name = os.getenv('MODEL_NAME')

prompt_template = ChatPromptTemplate.from_messages([
    ("system","Translate the following from English into {language}"),
    ("user", "{text}")
])

llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)

parser = StrOutputParser()

chain = prompt_template | llm | parser
response = chain.invoke({"text": "Nice to meet you.", "language": "Chinese"})
print(response)