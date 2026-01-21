from langchain.agents.middleware import wrap_tool_call
from langchain.tools import tool
from langchain.agents import create_agent

import os
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

load_dotenv()
api_key = os.getenv('API_KEY_DASHSCOPE')
base_url = os.getenv('BASE_URL')
model_name = os.getenv('MODEL_NAME')

model = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)

@tool(description="获取某个城市的天气")
def get_city_weather(city: str):
    """获取某个城市的天气
    Args:
        city: 具体城市
    """
    return f"城市{city}, 今天天气不错"

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )


agent = create_agent(
    tools=[get_city_weather],
    model=model,
    system_prompt="你是一个多功能助手，请优先使用工具完成任务",
    # middleware=[handle_tool_errors]
)

query = ("上海今天天气如何")
# response = agent.invoke({"input": {"query": "上海今天天气如何"}})
response = agent.stream(
    {"messages": [{ "role": "user", "content": "上海今天天气如何"}]},
    stream_mode="updates"
)

for chunk in response:
    for step, data in chunk.items():
        print(f"step: {step}")
        print(f"content: {data['messages'][-1].content_blocks}")
# print(response)
