from langchain_community.llms import Ollama
from langchain_core.chat_history import InMemoryChatMessageHistory

llm = Ollama(model="qwen2.5:7b")

history = InMemoryChatMessageHistory()

history.add_user_message("你是谁？")
aimessage = llm.invoke(history.messages)

print(aimessage)

history.add_message(aimessage)
history.add_user_message("请重复一次")
aimessage2 = llm.invoke(history.messages)
print(aimessage2)

history.add_message(aimessage2)

print("Chat history:")
for message in history.messages:
    print(f"{type(message).__name__}：{message}")