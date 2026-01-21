from langchain.tools import tool
import datetime
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
api_key = os.getenv('API_KEY_DASHSCOPE')
base_url = os.getenv('BASE_URL')
model_name = os.getenv('MODEL_NAME')

llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)

@tool
def get_current_date():
    """This function get current date"""
    return datetime.datetime.today().strftime('%Y-%m-%d')

# LLM bind tools
llm_with_tools = llm.bind_tools([get_current_date])
# tool container
all_tools = {"get_current_date": get_current_date}

query = "今天是几月几号"
messages = [query]
# 询问大模型。大模型会判断需要调用工具，并返回一个工具调用请求
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)
# print invoke tools
print(ai_msg.tool_calls)
if ai_msg.tool_calls:
    for tool_call in ai_msg.tool_calls:
        selected_tool = all_tools[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)

res = llm_with_tools.invoke(messages)
print(res.content)