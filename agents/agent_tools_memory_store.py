from langchain_classic import hub
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

tools = [retriever_tool]

prompt = hub.pull("hwchase17/openai-functions-agent")

store = {}

agent = create_tool_calling_agent(
    model,
    tools,
    prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)
# shared the same session id
response = agent_with_chat_history.invoke(
    {"input": "Hi，我的名字是Franco"},
    config={"configurable": {"session_id": "abc123"}}
)
print(response)

response = agent_with_chat_history.invoke(
    {"input": "我叫什么名字?"},
    config={"configurable": {"session_id": "abc123"}}
)
print(response)
