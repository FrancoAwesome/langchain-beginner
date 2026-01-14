from langchain_text_splitters import CharacterTextSplitter
from chapter1_index import documents

# splitter documents
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0, separator="/n/n", keep_separator=True)

segments = text_splitter.split_documents(documents)
print(segments)
for segment in segments:
    print(segment.page_content)
    print("-"*50)
