from dotenv import load_dotenv
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
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
        "question": "什么是Python？",
        "answer": "Python是一种编程语言"
    }
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # example list
    examples,
    # check language similarity
    OpenAIEmbeddings(),
    # VectorStore class
    Chroma,
    # example count
    k=1
)

question = "谁的寿命更长，穆罕默德阿里还是艾伦图灵?"
selected_examples = example_selector.get_selected_example({
    "question": question
})
print(f"最相似的示例：{question}")
for example in selected_examples:
    print("\n{example}")
    for k,v in example.items():
        print(f"{k}: {v}")