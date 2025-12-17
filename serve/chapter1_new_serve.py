import uvicorn
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from config.load_key import load_envkey

prompt_template = ChatPromptTemplate.from_messages([
    ("system","Translate the following from English into {language}"),
    ("user","{text}")
])

llm = ChatOpenAI(
    model=load_envkey()["model_name"],
    base_url=load_envkey()["base_url"],
    api_key=load_envkey()["api_key"],
)

parser = StrOutputParser()

chain = prompt_template | llm | parser

from fastapi import FastAPI
from langserve import add_routes

app = FastAPI(
    title="Test Agent",
    version="1.0",
    description="Test Agent"
)

# /docs same as swagger-ui
# /langchainDemo/playground it's the testing page
add_routes(
    app,
    chain,
    path='/langchainDemo'
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# import uvicorn
# uvicorn.run(app, host="0.0.0.0", port=8000)