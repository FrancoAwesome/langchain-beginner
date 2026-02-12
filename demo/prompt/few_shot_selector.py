from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
import dotenv
load_dotenv()

examples = [
    {
        "question": "谁的寿命更长，穆罕默德阿里还是艾伦图灵?",
        "answer":
            """
            这里需要跟进问题吗：是的。
            跟进：穆罕默德阿里去世时多大？
            中间答案：穆罕默德阿里去世时74岁。
            跟进：艾伦图灵去世时多大？
            中间答案：艾伦图灵去世时41岁。
            所以最终答案时：穆罕默德阿里。
            """
    },
    {
        "question": "什么是Python?",
        "answer": "Python是一种编程语言"
    },
    {
        "question": "目前电影票房第一名是谁?",
        "answer": "《阿凡达》的票房是 27.9 亿美元。《复仇者联盟 4:终局之战》的票房是 27.8 亿美元。因此,《阿凡达》的票房更高。"
    }
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # example list
    examples,
    # check language similarity
    DashScopeEmbeddings(dashscope_api_key=os.environ.get("API_KEY_DASHSCOPE")),
    # OpenAIEmbeddings(),
    # VectorStore class
    Chroma,
    # example count
    k=1
)

question = ("谁的寿命更长?")
selected_examples = example_selector.select_examples({
    "question": question
})
print(f"最相似的示例：{question}")
for example in selected_examples:
    print("\n{example}")
    for k,v in example.items():
        print(f"{k}: {v}")