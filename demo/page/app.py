
import streamlit as st
import pandas as pd
import numpy as np

# 设置页面标题
st.set_page_config(page_title="Streamlit 示例", layout="wide")

# 页面头部
st.title("📊 Streamlit 功能演示")
st.markdown("""
这是一个展示 Streamlit 常用功能的简单示例，包括文本展示、图表绘制以及交互控件。
""")

# 侧边栏控制面板
with st.sidebar:
    st.header("⚙️ 控制面板")
    show_data = st.checkbox("显示数据表", value=True)
    chart_type = st.selectbox("选择图表类型", ["折线图", "柱状图"])
    num_points = st.slider("数据点数量", min_value=10, max_value=100, value=50)

# 数据生成部分
data = pd.DataFrame({
    'x': range(num_points),
    'y': np.cumsum(np.random.randn(num_points))
})

# 显示数据表
if show_data:
    st.subheader("📋 随机生成的数据表")
    st.dataframe(data.style.highlight_max(axis=0))

# 图表展示区
st.subheader("📈 数据可视化")
if chart_type == "折线图":
    st.line_chart(data.set_index('x'))
else:
    st.bar_chart(data.set_index('x'))

# 用户输入交互
st.subheader("💬 用户交互")
user_input = st.text_input("请输入一些文字:", placeholder="在这里输入...")
if user_input:
    st.write(f"你输入的内容是: _{user_input}_")

# 文件上传模拟
uploaded_file = st.file_uploader("📁 上传一个XLSX文件", type=["xlsx"])
if uploaded_file is not None:
    df_uploaded = pd.read_excel(uploaded_file)
    print(df_uploaded)
    st.success("文件已成功加载！")
    st.dataframe(df_uploaded)
else:
    st.info("尚未上传任何文件。")

# 进度条与按钮演示
if st.button("🚀 开始模拟计算"):
    progress_bar = st.progress(0)
    for i in range(100):
        import time
        time.sleep(0.02)  # 模拟耗时操作
        progress_bar.progress(i + 1)
    st.balloons()
    st.success("🎉 计算已完成！")

# 尾部信息
st.divider()
st.caption("💡 此为 Streamlit 示例页面，展示了常见组件和基本功能。")
