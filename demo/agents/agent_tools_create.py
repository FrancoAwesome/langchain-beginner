import langchain_text_splitters
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from config.load_key import load_envkey

loader = WebBaseLoader("https://zh.wikipedia.org/wiki/%E7%8C%AB")
loader.requests_per_second = 1
docs = loader.load()

documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)

print(documents)

# 将网页文本转换为向量并存储
vector = FAISS.from_documents(documents, DashScopeEmbeddings(
    dashscope_api_key=load_envkey("API_KEY_DASHSCOPE")
))
retriever = vector.as_retriever()

print(retriever.invoke("猫的特征")[0])

retriever_tool = create_retriever_tool(
    retriever,
    name="wiki",
    description="维基百科",
)

model = ChatOpenAI(
    model=load_envkey('MODEL_NAME'),
    base_url=load_envkey('BASE_URL'),
    api_key=load_envkey('API_KEY_DASHSCOPE')
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

tools = [retriever_tool]

agent = create_tool_calling_agent(
    model,
    tools,
    prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

print(agent_executor.invoke({"input": "猫的特征?"}))