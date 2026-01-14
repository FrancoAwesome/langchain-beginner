from langchain_core.output_parsers import JsonOutputParser, XMLOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from config.load_key import load_envkey

model = ChatOpenAI(
    model=load_envkey('MODEL_NAME'),
    base_url=load_envkey('BASE_URL'),
    api_key=load_envkey('API_KEY_DASHSCOPE')
)

joke_query = "生成周星驰的简化电影列表，按照最新时间的降序"
# format result to defined model class
# parser = XMLOutputParser()
# enhanced tags
parser = XMLOutputParser(tags=["movies", "actor", "film", "name", "genre"])
prompt = PromptTemplate(
    template="回答用户的查询，\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
chain = prompt | model | parser
response = chain.invoke({"query": joke_query})
print(response)