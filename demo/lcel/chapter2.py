import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnableLambda

load_dotenv()
api_key = os.getenv('API_KEY_DASHSCOPE')
base_url = os.getenv('BASE_URL')
model_name = os.getenv('MODEL_NAME')

llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)

parser = StrOutputParser()
# create prompt template for Chinese
prompt_template_zh = ChatPromptTemplate.from_messages([
    ("system","Tran[late the following from English into Chinese"),
    ("user", "{text}")
])
# create prompt template for French
prompt_template_fr = ChatPromptTemplate.from_messages([
    ("system","Translate the following from English into French"),
    ("user", "{text}")
])
# create chain for Chinese and French
chain_zh = prompt_template_zh | llm | parser
chain_fr = prompt_template_fr | llm | parser
# parallel run 2 chains
parallel_chains = RunnableMap({
    "zh-translation": chain_zh,
    "fr-translation": chain_fr
})

# merge responses
final_chain = parallel_chains | RunnableLambda(lambda x: f"Chinese: {x['zh-translation']}\nFrench: {x['fr-translation']}")
final_chain.get_graph().print_ascii();

# invoke chain
print(final_chain.invoke({"text": "nice to meet you"}))
