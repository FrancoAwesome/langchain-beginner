import os
import tempfile
from dotenv import load_dotenv

import streamlit as st
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from rag_multimodal import RAGMultimodal

target_directory = "./resources/heros3"

def initialize_app():
    load_dotenv()
    embedding_model = os.getenv("EMBEDDING_MODEL")
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGMultimodal(resources_dir=target_directory, embedding_model=embedding_model)
        with st.spinner('初始化知识库...'):
            success = st.session_state.rag_system.initialize_knowledge_base()
            if success:
                st.success('知识库初始化成功！')
            else:
                st.error('知识库初始化失败，可能是resources目录为空。')
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def get_llm_response(query, context):
    api_key = os.getenv("API_KEY_DASHSCOPE")
    if not api_key:
        return "错误：API_KEY_DASHSCOPE 环境变量未设置"
    
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY_DASHSCOPE"),
    )
    
    prompt = ChatPromptTemplate.from_template("""
    你是一个智能助手，需要根据提供的上下文信息回答用户的问题。
    上下文信息可能包含文本和图片描述。
    请根据上下文信息，用自然、友好的语言回答用户的问题。
    如果上下文信息不足以回答问题，请如实告知。
    
    上下文信息：
    {context}
    
    用户问题：
    {query}
    """)

    chain = prompt | llm | StrOutputParser()
    try:
        response = chain.invoke({
            "query": query,
            "context": query + "\n" + context,
        })
        print(f"response: {response}")
        return response
    except Exception as e:
        return f"错误：{str(e)}"

def get_chat_history():
    if st.session_state.chat_history:
        for speaker, message in st.session_state.chat_history:
            if speaker == "用户":
                st.markdown(f"**👤 你：** {message}")
            else:
                st.markdown(f"**🤖 AI：** {message}")

def main():
    st.set_page_config(
        page_title="多模态RAG知识库",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("多模态RAG知识库系统")
    st.write("基于DashScope和FAISS的多模态文档检索与问答系统")
    
    # 初始化应用
    initialize_app()
    
    # 侧边栏：上传文档
    with st.sidebar:
        st.header("扩展英雄无敌3知识库")
        uploaded_files = st.file_uploader(
            "上传文档（支持txt、pdf、jpg、jpeg、png、gif）",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                print(f"uploaded file: {uploaded_file}")
                with st.spinner(f"处理文件：{uploaded_file.name}..."):
                    # 创建临时文件
                    # with tempfile.NamedTemporaryFile(
                    #     encoding="utf-8",
                    #     delete=False,
                    #     suffix=os.path.splitext(uploaded_file.name)[1]
                    # ) as tmp_file:
                    #     tmp_file.write(uploaded_file.getvalue())
                    #     tmp_file_path = tmp_file.name

                    # 构建完整的文件路径
                    tmp_file_path = os.path.join(target_directory, uploaded_file.name)

                    # 保存临时文件
                    with open(tmp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    print(f"tmp file path: {tmp_file_path}")
                    # 添加文档到知识库
                    success, message = st.session_state.rag_system.add_document(tmp_file_path)
                    print("uploaded")
                    # 删除临时文件
                    os.unlink(tmp_file_path)
                    
                    if success:
                        st.success(f"文件 {uploaded_file.name} 上传成功！{message}")
                    else:
                        st.error(f"文件 {uploaded_file.name} 上传失败：{message}")
    
    # 主界面：查询知识库
    st.header("查询英雄无敌3知识库")
    query = st.text_input("请输入您的问题：", placeholder="例如：请介绍英雄无敌3城堡7级兵？")
    
    if query:
        with st.spinner('搜索知识库...'):
            # 在向量数据库中搜索最相似的2个文档
            results, message = st.session_state.rag_system.query_knowledge_base(query, k=2)
            print(f"results: {results}")
            
            if results:
                st.subheader("搜索结果")
                
                # 构建上下文
                context = "\n".join([doc.page_content for doc in results])
                print(context)
                
                # 获取LLM回答
                with st.spinner('生成回答...'):
                    llm_response = get_llm_response(query, context)
                
                st.subheader("AI回答")
                st.write(llm_response)
                # store the chat history for both user and AI
                st.session_state.chat_history.append(("用户", query))
                st.session_state.chat_history.append(("AI", llm_response))

                st.markdown("-"*50)
                st.subheader("聊天记录")
                get_chat_history()
                
                # st.subheader("相关文档")
                #
                # for i, doc in enumerate(results, 1):
                #     st.write(f"### 文档 {i}")
                #     st.write(doc.page_content)
                #     st.write(f"**来源：** {doc.metadata.get('source', '未知')}")
                #     st.write("---")
            else:
                st.error(f"搜索失败：{message}")

if __name__ == "__main__":
    main()