from langchain_core.tools import StructuredTool, ToolException

def get_weather(city: str) -> int:
    """city"""
    raise ToolException(f"没有名为:{city}的城市")

def handle_weather_error(error: ToolException):
    """city"""
    return f"工具执行期间发生错误：{error.args[0]}"

get_weather_tool = StructuredTool.from_function(
    func=get_weather,
    error_handler=handle_weather_error,
    return_result=True
)

response = get_weather_tool.invoke({"city": "abc"})
print(response)