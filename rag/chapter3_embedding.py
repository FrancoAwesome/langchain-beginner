from langchain_community.embeddings import DashScopeEmbeddings
from config.load_key import load_envkey
from chapter2_splitter import segments

embedding_model = DashScopeEmbeddings(
    dashscope_api_key=load_envkey('API_KEY_DASHSCOPE'),
    model="text-embedding-v1"
)

redis_url = "redis://localhost:6379"

# from langchain_redis import RedisConfig, RedisVectorStore
# config = RedisConfig(
#     index_name="meituan-index",
#     redis_url=redis_url
# )
#
# vector_store = RedisVectorStore(embedding_model, config=config)

from langchain_chroma import Chroma
vector_store = Chroma(embedding_function=embedding_model)

vector_store.add_documents(segments)

print(vector_store)