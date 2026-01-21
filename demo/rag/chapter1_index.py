from langchain_community.document_loaders import TextLoader, DirectoryLoader

loader = TextLoader("../../resources/meituan-questions.txt", encoding="utf-8")
# loader = TextLoader("../resources/test/test.txt")
documents = loader.load()
print(documents)

# directorLoader = DirectoryLoader("../resources/test/", glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
# documents = directorLoader.load()
# print(documents)