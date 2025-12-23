from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_redis import RedisChatMessageHistory
from config.load_key import load_envkey

prompt = ChatPromptTemplate.from_messages([
    ("system","You're an assistant who's good at {ability}. Respnd in 20 words or fewer"),
    MessagesPlaceholder(variable_name="history"), # history message placeholder
    ("user", "{input}")
])
model = ChatOpenAI(
    model=load_envkey('MODEL_NAME'),
    base_url=load_envkey('BASE_URL'),
    api_key=load_envkey('API_KEY_DASHSCOPE')
)
runnable = prompt | model

store = {}

redis_url = "redis://localhost:6379/0"

def get_session_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(session_id=session_id, redis_url=redis_url)

with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)
# shared the same session id
response = with_message_history.invoke(
    {"ability": "math", "input": "余弦是什么意思？"},
    config={"configurable": {"session_id": "abc123"}}
)
print(response)

# not shared the same session id, no memory
response = with_message_history.invoke(
    {"ability": "math", "input": "什么？"},
    config={"configurable": {"session_id": "abc123"}}
)
print(response)

response = with_message_history.invoke(
    {"ability": "math", "input": "什么？"},
    config={"configurable": {"session_id": "def234"}}
)
print(response)