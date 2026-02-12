import os
import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain_community.embeddings import DashScopeEmbeddings

chinese_separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

class RAGMultimodal:
    def __init__(self, resources_dir="./resources", embedding_model="text-embedding-v2"):
        self.resources_dir = resources_dir
        self.embedding_model = embedding_model
        self.vector_store = None
        self.embeddings = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            # separators=chinese_separators
        )
        # self.text_splitter = CharacterTextSplitter(
        #     chunk_size=500,
        #     chunk_overlap=100,
        #     separator="\n\n",
        #     keep_separator=True
        # )
    def initialize_embeddings(self):
        if self.embeddings is None:
            api_key = os.getenv("API_KEY_DASHSCOPE")
            if not api_key:
                raise ValueError("API_KEY_DASHSCOPE environment variable not set")
            self.embeddings = DashScopeEmbeddings(
                model=self.embedding_model,
                dashscope_api_key=api_key
            )
    def load_documents(self):
        documents = []
        
        text_loader = DirectoryLoader(
            self.resources_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            recursive=True,
            loader_kwargs={"encoding": "utf-8"}
        )
        text_docs = text_loader.load()
        documents.extend(text_docs)
        
        pdf_loader = DirectoryLoader(
            self.resources_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            recursive=True,
            loader_kwargs={"encoding": "utf-8"}
        )
        pdf_docs = pdf_loader.load()
        documents.extend(pdf_docs)
        
        image_loader = DirectoryLoader(
            self.resources_dir,
            glob="**/*.{jpg,jpeg,png,gif}",
            loader_cls=UnstructuredImageLoader,
            recursive=True
        )
        image_docs = image_loader.load()
        documents.extend(image_docs)
        
        return documents
    def initialize_knowledge_base(self):
        self.initialize_embeddings()
        
        documents = self.load_documents()
        if not documents:
            print("No documents found in resources directory")
            return False
        
        split_docs = self.text_splitter.split_documents(documents)
        
        self.vector_store = FAISS.from_documents(
            split_docs,
            self.embeddings
        )
        
        print(f"Knowledge base initialized with {len(split_docs)} document chunks")
        return True
    def add_document(self, file_path):
        self.initialize_embeddings()

        print(f"file existed: {self.check_file_exists(file_path)}")

        if not self.check_file_exists(file_path):
            return False, "File not existed"


        if file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif any(file_path.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
            loader = UnstructuredImageLoader(file_path)
        else:
            return False, "Unsupported file format"
        
        try:
            docs = loader.load()
            split_docs = self.text_splitter.split_documents(docs)
            
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(
                    split_docs,
                    self.embeddings
                )
            else:
                self.vector_store.add_documents(split_docs)
            
            return True, f"Document added successfully. Added {len(split_docs)} chunks."
        except Exception as e:
            return False, f"Error adding document: {str(e)}"
    def query_knowledge_base(self, query, k=5):
        if self.vector_store is None:
            return None, "Knowledge base not initialized"
        
        try:
            results = self.vector_store.similarity_search(
                query=query,
                k=k
            )
            score = self.get_similarity_score(query, results)
            return results, "Query successful"
        except Exception as e:
            return None, f"Error querying knowledge base: {str(e)}"
    def get_similarity_score(self, query, doc_content):
        self.initialize_embeddings()
        
        try:
            query_embedding = self.embeddings.embed_query(query)
            doc_embedding = self.embeddings.embed_query(doc_content)
            
            if len(query_embedding) != len(doc_embedding):
                return 0.0
            
            dot_product = np.dot(query_embedding, doc_embedding)
            query_norm = np.linalg.norm(query_embedding)
            doc_norm = np.linalg.norm(doc_embedding)
            
            if query_norm == 0 or doc_norm == 0:
                return 0.0
            
            similarity = dot_product / (query_norm * doc_norm)
            return similarity
        except Exception as e:
            return 0.0

    def check_file_exists(self, file_path):
        """
        判断文件是否存在

        Args:
            file_path (str): 文件路径

        Returns:
            bool: 文件存在返回True，否则返回False
        """
        return os.path.exists(file_path)
