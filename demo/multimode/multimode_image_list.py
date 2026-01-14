from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
# get base64 code of image
import base64
# get ackii of image
import httpx
from config.load_key import load_envkey

image_url1 = "https://img1.baidu.com/it/u=3358136179,3360223400&fm=253&app=120&f=JPEG?w=1422&h=800"
image_url2 = "https://img1.baidu.com/it/u=722361187,823811711&fm=253&app=120&f=JPEG?w=1422&h=800"
model = ChatOpenAI(
    model=load_envkey('MODEL_NAME'),
    base_url=load_envkey('BASE_URL'),
    api_key=load_envkey('API_KEY_DASHSCOPE')
)

message = HumanMessage(
    content=[
        {"type": "text", "text": "这两张图片是一样的吗？"},
        # using image url but will face the condition that agent have no access for the image url
        {"type": "image_url", "image_url": {"url": image_url1}},
        {"type": "image_url", "image_url": {"url": image_url2}}
    ]
)

response = model.invoke([message])
print(response.content)