import langchain_text_splitters
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

loader = WebBaseLoader("https://zh.wikipedia.org/wiki/%E7%8C%AB")
loader.requests_per_second = 1
docs = loader.load()

documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)

print(documents)

# 将网页文本转换为向量并存储
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()

print(retriever.invoke("猫的特征")[0])

retriever_tool = create_retriever_tool(
    retriever,
    name="wiki",
    description="维基百科",
)




