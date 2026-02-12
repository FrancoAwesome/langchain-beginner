import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from operator import itemgetter
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()
base_url = os.getenv('BASE_URL')
model_name = os.getenv('MODEL_NAME')
api_key = os.getenv("API_KEY_DASHSCOPE")

print(api_key)

def collect_document(segments):
    text = []
    for segment in segments:
        text.append(segment.page_content)
    return text

def load_single_document(path,chunk_size = 200, chunk_overlap = 10):
    loader = TextLoader(f"{path}/城堡.txt", encoding="utf-8")
    documents = loader.load()
    return documents

def load_documents(path, chunk_size = 200, chunk_overlap = 10, is_upload: bool = False):
    director_loader = DirectoryLoader(
        path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True)
    documents = director_loader.load()
    return documents

def store_documents(path, chunk_size = 200, chunk_overlap = 10, is_single: bool = False):
    print("加载文档...")
    if is_single:
        documents = load_single_document(path)
    else:
        documents = load_documents(path)

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator="/n/n",
                                          keep_separator=True)
    segments = text_splitter.split_documents(documents)
    print(segments)
    # 创建向量存储
    embeddings = DashScopeEmbeddings(dashscope_api_key=api_key, model="text-embedding-v1")
    # st.session_state.knowledge_base = FAISS.from_documents(segments, embeddings)
    st.session_state.knowledge_base = FAISS.from_documents(segments, embeddings)
    print("文档加载完成")

prompt_template = ChatPromptTemplate.from_messages([
    ("user", """你是一个答疑机器人，你的任务是根据下述给定的已知信息回答用户的问题
    已知信息：{context}
    用户问题：{question}
    如果已知信息不包含用户问题的答案，或者已知信息不足以回答用户的问题，请直接回复“我无法回答你的问题。
    请不要输出已知信息中不包含的信息或答案。
    请用中文回答用户问题。""")
])

st.set_page_config("魔法门之英雄无敌3")

st.title("英雄无敌3")
st.markdown("""
    这是一个提供最官方最全面的关于英雄无敌3的数据，包含各种族介绍，以及各种族兵种的介绍以及优劣势。
            """)

# 初始化session_state
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("---")
user_question = st.text_input("请输入您的问题：", placeholder="例如：英雄无敌3中城堡兵种介绍？")
ask_button = st.button("提交问题")
# 如果没有知识库，初始化知识库
if not st.session_state.knowledge_base:
    store_documents("../../resources/heros3/")

print(f"user input: {user_question}")
if ask_button and user_question and st.session_state.knowledge_base:
    with st.spinner("正在生成答案..."):
        # 搜索相关文档片段
        # docs = st.session_state.knowledge_base.as_retriever()
        retriever = st.session_state.knowledge_base.as_retriever(search_type="mmr", search_kwargs={"k": 5})
        print(f"docs: {retriever}")

        # 使用ChatOpenAI问答链
        llm = ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key)
        chain = {
            "context": itemgetter("question") | retriever | collect_document,
            "question": itemgetter("question")
        } | prompt_template | llm | StrOutputParser()

        response = chain.invoke({"question": user_question})

        # 保存对话历史
        st.session_state.chat_history.append(("用户", user_question))
        st.session_state.chat_history.append(("AI", response))
        st.balloons()
        st.success("🎉 查询已完成！")

# 显示对话历史
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("💬 对话历史")
    for speaker, message in st.session_state.chat_history:
        if speaker == "用户":
            st.markdown(f"**👤 你：** {message}")
        else:
            st.markdown(f"**🤖 AI：** {message}")

