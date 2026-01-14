from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

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

example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="问题：{question}\n{answer}"
)

# prompt = FewShotPromptTemplate(
#     examples=examples,
#     examples_prompt=examples_prompt,
#     prefix="你是一个AI助手，请根据以下示例回答问题：",
#     suffix="问题：{input}\\n回答：",
#     input_variables=["input"],
# )
#
# print(prompt.format(input="谁的寿命更长，穆罕默德艾莉还是艾伦图灵"))

# 创建FewShotPromptTemplate实例
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="你是一个AI助手，请根据以下示例回答问题：",
    suffix="问题: {input}\n",
    input_variables=["input"]
)

# 格式化提示词
formatted_prompt = few_shot_prompt.format(input="谁的寿命更长，穆罕默德阿里还是艾伦图灵?")
print(formatted_prompt)