from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig, ConfigurableFieldSpec
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from config.load_key import load_envkey

prompt = ChatPromptTemplate.from_messages([
    ("system","You're an assistant who's good at {ability}. Respnd in 20 words or fewer"),
    MessagesPlaceholder(variable_name="history"), # history message place holder
    ("user", "{input}")
])
model = ChatOpenAI(
    model=load_envkey('MODEL_NAME'),
    base_url=load_envkey('BASE_URL'),
    api_key=load_envkey('API_KEY_DASHSCOPE')
)
runnable = prompt | model

store = {}

def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = ChatMessageHistory()
    return store[(user_id, conversation_id)]

with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation="str",
            name="User Id",
            description="The id of the user",
            default=None,
            is_shared=True
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation="str",
            name="Conversation Id",
            description="The id of the conversation",
            default=None,
            is_shared=True
        )
    ]
)
# shared the same session id
response = with_message_history.invoke(
    {"ability": "math", "input": "余弦是什么意思？"},
    config={"configurable": {"user_id": "123", "conversation_id": "1"}}
)
print(response)

# not shared the same session id, no memory
response = with_message_history.invoke(
    {"ability": "math", "input": "什么？"},
    config={"configurable": {"user_id": "123", "conversation_id": "1"}}
)
print(response)

response = with_message_history.invoke(
    {"ability": "math", "input": "什么？"},
    config={"configurable": {"user_id": "234", "conversation_id": "2"}}
)
print(response)
