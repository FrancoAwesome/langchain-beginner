from langchain_core.output_parsers import StrOutputParser
import chapter3_embedding

query = "在线支付取消订单后钱怎么返还"

from langchain_community.embeddings import DashScopeEmbeddings
from config.load_key import load_envkey
import os

if not os.environ.get("DASHSCOPE_API_KEY"):
    os.environ["DASHSCOPE_API_KEY"] = load_envkey("API_KEY_DASHSCOPE")

embedding_model = DashScopeEmbeddings(model="text-embedding-v1")

from langchain_chroma import Chroma

vector_store = Chroma(embedding_function=embedding_model)
retriever = vector_store.as_retriever()

# results= vector_store.similarity_search(query, k=2)
# print(results)

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model=load_envkey("MODEL_NAME"),
    base_url=load_envkey("BASE_URL"),
    api_key=load_envkey("API_KEY_DASHSCOPE"),
)

from langchain_core.prompts import ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_messages([
    ("user", """你是一个答疑机器人，你的任务是根据下述给定的已知信息回答用户的问题
    已知信息：{context}
    用户问题：{question}
    如果已知信息不包含用户问题的答案，或者已知信息不足以回答用户的问题，请直接回复“我无法回答你的问题。
    请不要输出已知信息中不包含的信息或答案。
    请用中文回答用户问题。""")
])

def collect_document(segments):
    text = []
    for segment in segments:
        text.append(segment.page_content)
    return text

from operator import itemgetter
chain = ({
    "context": itemgetter("question") | retriever | collect_document,
    "question": itemgetter("question")
    }
    | prompt_template | llm | StrOutputParser()
)

response = chain.invoke({"question": query})
print(response)
