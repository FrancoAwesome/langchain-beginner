from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config.load_key import load_envkey

model = ChatOpenAI(
    model=load_envkey('MODEL_NAME'),
    base_url=load_envkey('BASE_URL'),
    api_key=load_envkey('API_KEY_DASHSCOPE')
)

class Joke(BaseModel):
    setup: str = Field(description="设置笑话的问题")
    punchline: int = Field(description="解决笑话的答案")

joke_query = "告诉我一个笑话，小于50字"
# format result to defined model class
parser = JsonOutputParser(pydantic_object=Joke)
# no format result
# parser = JsonOutputParser()
prompt = PromptTemplate(
    template="回答用户的查询，\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
chain = prompt | model | parser
response = chain.invoke({"query": joke_query})
print(response)