from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
# get base64 code of image
import base64
# get ackii of image
import httpx
from config.load_key import load_envkey

image_url = "https://img1.baidu.com/it/u=3358136179,3360223400&fm=253&app=120&f=JPEG?w=1422&h=800"
image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
model = ChatOpenAI(
    model=load_envkey('MODEL_NAME'),
    base_url=load_envkey('BASE_URL'),
    api_key=load_envkey('API_KEY_DASHSCOPE')
)

message = HumanMessage(
    content=[
        {"type": "text", "text": "用中文描述这张图片中的天气"},
        # using base64 encode of image to avoid the condition that agent have no access for the image url
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
    ]
)

response = model.invoke([message])
print(response.content)