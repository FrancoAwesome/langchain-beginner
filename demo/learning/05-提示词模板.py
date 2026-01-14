# from langchain_core.prompts import PromptTemplate
#
# prompt = PromptTemplate.from_template('你号我是{}')
#


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个专业的技术文档编写者'),
    ('user', '请写一篇关于{xyz}的文档')
])

print(prompt.invoke({'xyz':'python'}))

