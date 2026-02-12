import os
import base64
from operator import itemgetter
from typing import List, Dict, Any

from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from PIL import Image
import io
import tempfile

from langchain_text_splitters import RecursiveCharacterTextSplitter


class MultimodalRAG:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """初始化多模态RAG系统"""
        self.embeddings = DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=os.environ.get("API_KEY_DASHSCOPE"))
        self.llm = ChatOpenAI(model="qwen-plus", api_key=os.environ.get("API_KEY_DASHSCOPE"))
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def add_text_document(self, content: str, metadata: Dict[str, Any] = None) -> None:
        """添加文本文档到知识库"""
        docs = self.text_splitter.create_documents([content], metadatas=[metadata] if metadata else [{}])
        self.vectorstore.add_documents(docs)

    def add_image_document(self, image_path: str, description: str = "", metadata: Dict[str, Any] = None) -> None:
        """添加图像文档到知识库"""
        # 读取图像并生成描述
        try:
            image = Image.open(image_path)
            # 这里可以集成图像识别模型来生成描述
            # 为简化示例，使用传入的描述
            content = f"图像描述: {description}"
            img_metadata = metadata or {}
            img_metadata.update({
                "image_path": image_path,
                "doc_type": "image"
            })
            doc = Document(page_content=content, metadata=img_metadata)
            self.vectorstore.add_documents([doc])
        except Exception as e:
            print(f"添加图像文档失败: {e}")

    def add_pdf_document(self, pdf_path: str, metadata: Dict[str, Any] = None) -> None:
        """添加PDF文档到知识库"""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            text_content = ""
            for page in reader.pages:
                text_content += page.extract_text()

            docs = self.text_splitter.create_documents([text_content], metadatas=[metadata] if metadata else [{}])
            self.vectorstore.add_documents(docs)
        except Exception as e:
            print(f"添加PDF文档失败: {e}")

    def query(self, question: str) -> str:
        """查询知识库并生成回答"""
        # 定义提示词模板
        prompt_template = """使用以下上下文回答问题，如果不知道答案，请说不知道。
        上下文: {context}
        问题: {question}
        答案:"""
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        retriever = self.vectorstore.as_retriever()
        # 创建RAG链

        chain = {
                    "context": itemgetter("question") | retriever,
                    "question": itemgetter("question")
                } | prompt | self.llm | StrOutputParser()


        # 执行查询
        result = chain.invoke({"question": question})
        return result["result"]


def main():
    """主函数示例"""
    # 初始化RAG系统
    rag = MultimodalRAG()

    # 添加文本文档
    text_content = """
    人工智能是计算机科学的一个分支，它企图了解智能的实质，
    并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    """
    rag.add_text_document(text_content, {"source": "ai_intro", "type": "text"})

    # 添加PDF文档（需要实际PDF文件）
    # rag.add_pdf_document("sample.pdf", {"source": "sample_pdf"})

    # 查询示例
    query_text = "人工智能包括哪些研究领域？"
    answer = rag.query(query_text)
    print(f"问题: {query_text}")
    print(f"答案: {answer}")


if __name__ == "__main__":
    main()
