import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
# 携带聊天记录
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()
api_key = os.getenv('API_KEY_DASHSCOPE')
base_url = os.getenv('BASE_URL')
model_name = os.getenv('MODEL_NAME')

llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)
prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个乐于助人的助手，尽你所能回答{data}问题。'),
    ('user', '{message_key}')
])

chain = prompt | llm
store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 实现保存历史聊天记录
do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='message_key')

config={'configurable': {'session_id': '1001'}}

# 第一轮
# res = do_message.invoke({
#     'message_key': '你好，我是扣扣',
#     'data': 'python'
# },
#     config=config
# )
#
# print(res)
#
# # 第二轮
# res = do_message.invoke({
#     'message_key': '你好，我是谁',
#     'data': 'python'
# },
#     config=config
# )
# print(res)

# 流式输出
resp2 = do_message.stream({
    'message_key': '你好，python之父是谁?',
    'data': 'python'
},
    config=config
)

# print(resp2)
for i in resp2:
    print(i.content)
