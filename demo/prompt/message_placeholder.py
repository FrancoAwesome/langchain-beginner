from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant!"),
    MessagesPlaceholder("msgs")
])

result = prompt_template.invoke({
    "msgs": [HumanMessage(content="hi!"), HumanMessage(content="hello!")]
})
print(result)