from langchain_core.prompts import load_prompt
# 加载yaml提示词模板
prompt = load_prompt("ft.yaml", "utf-8")
print(prompt.format(name="狐狸", what= "喜剧"))
# 加载json提示词模板
prompt = load_prompt("ft.json", "utf-8")
print(prompt.format(name="狐狸", what= "喜剧"))