import streamlit as st
import requests
import os

baseUrl = os.getenv("OLLAMA_BASE_URL")

# Ollama API配置
OLLAMA_API_URL = f"{baseUrl}/api/generate"
MODEL_NAME = "qwen2.5:7b"

# 模拟文档库（实际项目中可替换为真实数据源）
DOCUMENTS = [
    "Streamlit是一个用于机器学习和数据科学的开源Python库，可以快速创建Web应用。",
    "Ollama是一个用于在本地运行大型语言模型的工具，支持多种开源模型。",
    "RAG(Retrieval-Augmented Generation)是一种结合信息检索和文本生成的技术。",
    "Python是一种高级编程语言，广泛用于数据科学、人工智能和Web开发。",
    "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式。"
]


def retrieve_context(query, documents, top_k=3):
    """简单的关键词匹配检索函数"""
    scores = []
    query_words = set(query.lower().split())

    for i, doc in enumerate(documents):
        doc_words = set(doc.lower().split())
        # 计算交集词数作为相似度得分
        score = len(query_words.intersection(doc_words))
        scores.append((score, doc, i))

    # 按得分排序并返回前top_k个文档
    scores.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc, _ in scores[:top_k]]


def query_ollama(prompt, model=MODEL_NAME):
    """向Ollama API发送请求"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "")
        else:
            return f"API请求失败，状态码: {response.status_code}"
    except Exception as e:
        return f"请求异常: {str(e)}"


def main():
    st.set_page_config(
        page_title="RAG智能体",
        page_icon="🤖",
        layout="wide"
    )

    # 页面标题和描述
    st.title("🤖 RAG智能体演示")
    st.markdown("""
    基于Streamlit和Ollama的检索增强生成(RAG)问答系统
    - 使用本地LLM模型进行推理
    - 通过关键词匹配检索相关上下文
    """)

    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 配置选项")
        model_name = st.selectbox(
            "选择模型",
            ["qwen2.5:7b", "deepseek-r1:7b", "deepseek-r1:1.5b"],
            index=0
        )

        top_k = st.slider(
            "检索文档数量",
            min_value=1,
            max_value=5,
            value=3
        )

        st.markdown("---")
        st.markdown("### 📚 文档库")
        for i, doc in enumerate(DOCUMENTS):
            st.caption(f"{i + 1}. {doc[:50]}...")

    # 主界面
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💬 问答区域")
        query = st.text_input(
            "请输入你的问题：",
            placeholder="例如：什么是RAG技术？",
            key="query_input"
        )

        if st.button("🚀 提交问题", use_container_width=True) or query:
            if not query.strip():
                st.warning("请输入有效问题")
            else:
                with st.spinner("正在检索相关文档..."):
                    # 检索相关上下文
                    context_docs = retrieve_context(query, DOCUMENTS, top_k)
                    context_text = "\n".join(context_docs)

                with st.spinner("正在生成答案..."):
                    # 构造提示词
                    prompt = f"""
                    基于以下上下文回答问题。如果上下文中没有相关信息，请说明无法基于提供的文档回答该问题。

                    上下文：
                    {context_text}

                    问题：{query}

                    回答：
                    """

                    # 调用Ollama模型
                    answer = query_ollama(prompt, model_name)

                # 显示答案
                st.subheader("📝 答案")
                st.info(answer)

    with col2:
        st.subheader("📄 检索到的文档")
        if 'context_docs' in locals():
            for i, doc in enumerate(context_docs):
                st.markdown(f"**文档 {i + 1}**")
                st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>{doc}</div>",
                            unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.info("提交问题后将显示相关文档")

    # 系统状态检查
    st.markdown("---")
    st.subheader("🔧 系统状态")

    col3, col4 = st.columns(2)
    with col3:
        try:
            response = requests.get(f"{baseUrl}/api/tags")
            if response.status_code == 200:
                st.success("✅ Ollama服务运行正常")
                print(response.json())
                models = response.json().get("models", [])
                st.markdown(f"**可用模型数量**: {len(models)}")
            else:
                st.error("❌ Ollama服务连接失败")
        except:
            st.error("❌ 无法连接到Ollama服务，请确保已启动")

    with col4:
        st.markdown("**文档库状态**")
        st.markdown(f"- 文档总数: {len(DOCUMENTS)}")
        st.markdown("- 检索方式: 关键词匹配")


if __name__ == "__main__":
    main()
