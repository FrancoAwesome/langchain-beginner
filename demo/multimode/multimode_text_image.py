import os
import base64
import json
import numpy as np
from typing import List, Dict, Any
import dashscope
from dashscope import Generation
from http import HTTPStatus
import requests
from io import BytesIO
from PIL import Image
import uuid


class DashScopeMultimodalRAG:
    def __init__(self, api_key: str):
        """
        初始化DashScope多模态RAG系统
        :param api_key: DashScope API密钥
        """
        self.api_key = api_key
        dashscope.api_key = api_key
        self.documents = []  # 存储文档信息
        self.embeddings = {}  # 存储文档嵌入向量

    def encode_image_to_base64(self, image_path: str) -> str:
        """
        将本地图片转换为base64编码
        :param image_path: 图片路径
        :return: base64编码字符串
        """
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
        except Exception as e:
            print(f"图片编码错误: {e}")
            return ""

    def get_multimodal_embedding(self, content: List[Dict]) -> List[float]:
        """
        使用DashScope获取多模态嵌入向量
        :param content: 包含文本和图片的内容列表
        :return: 嵌入向量
        """
        try:
            response = dashscope.TextEmbedding.call(
                model='text-embedding-v2',
                input=json.dumps(content)
            )

            if response.status_code == HTTPStatus.OK:
                embedding = response.output['embeddings'][0]['embedding']
                return embedding
            else:
                print(f'获取嵌入失败: {response.message}')
                # 返回零向量作为默认值
                return [0.0] * 1536

        except Exception as e:
            print(f"获取嵌入时发生异常: {e}")
            return [0.0] * 1536

    def add_document(self, doc_id: str, text_content: str = "", image_path: str = None,
                     image_url: str = None, metadata: Dict = None):
        """
        添加文档到知识库
        :param doc_id: 文档ID
        :param text_content: 文本文档内容
        :param image_path: 本地图片路径
        :param image_url: 在线图片URL
        :param metadata: 元数据
        """
        content_items = []

        # 添加文本内容
        if text_content:
            content_items.append({
                "text": text_content
            })

        # 添加图片内容
        if image_path:
            image_base64 = self.encode_image_to_base64(image_path)
            if image_base64:
                content_items.append({
                    "image": f"data:image/jpeg;base64,{image_base64}"
                })
        elif image_url:
            content_items.append({
                "image": image_url
            })

        if not content_items:
            print("文档至少需要包含文本或图片内容")
            return

        # 保存文档信息
        document = {
            "id": doc_id,
            "content_items": content_items,
            "text_content": text_content,
            "image_path": image_path,
            "image_url": image_url,
            "metadata": metadata or {}
        }

        self.documents.append(document)
        print(f"已添加文档 ID: {doc_id}")

    def build_index(self):
        """
        为所有文档构建索引
        """
        print("开始构建文档索引...")
        for i, doc in enumerate(self.documents):
            print(f"处理文档 {i + 1}/{len(self.documents)} (ID: {doc['id']})")
            embedding = self.get_multimodal_embedding(doc["content_items"])
            self.embeddings[doc["id"]] = embedding
        print("索引构建完成!")

    def calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        :param vec1: 第一个向量
        :param vec2: 第二个向量
        :return: 余弦相似度
        """
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0

        return dot_product / (norm_vec1 * norm_vec2)

    def search(self, query_text: str = "", query_image_path: str = None,
               query_image_url: str = None, top_k: int = 3) -> List[Dict]:
        """
        多模态搜索
        :param query_text: 查询文本
        :param query_image_path: 查询本地图片路径
        :param query_image_url: 查询在线图片URL
        :param top_k: 返回最相似的K个结果
        :return: 搜索结果列表
        """
        if not query_text and not query_image_path and not query_image_url:
            print("查询至少需要包含文本或图片")
            return []

        # 构造查询内容
        query_items = []
        if query_text:
            query_items.append({"text": query_text})
        if query_image_path:
            image_base64 = self.encode_image_to_base64(query_image_path)
            if image_base64:
                query_items.append({
                    "image": f"data:image/jpeg;base64,{image_base64}"
                })
        elif query_image_url:
            query_items.append({"image": query_image_url})

        if not query_items:
            print("无法构造有效的查询内容")
            return []

        # 获取查询嵌入向量
        print("获取查询嵌入向量...")
        query_embedding = self.get_multimodal_embedding(query_items)

        # 计算相似度
        print("计算相似度...")
        similarities = []
        for doc_id, doc_embedding in self.embeddings.items():
            similarity = self.calculate_cosine_similarity(query_embedding, doc_embedding)
            similarities.append((doc_id, similarity))

        # 排序并返回Top-K结果
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similarities = similarities[:top_k]

        # 构造返回结果
        results = []
        for doc_id, score in top_similarities:
            # 查找原始文档
            doc = next((d for d in self.documents if d["id"] == doc_id), None)
            if doc:
                result = {
                    "document_id": doc_id,
                    "similarity_score": score,
                    "text_content": doc["text_content"],
                    "image_path": doc["image_path"],
                    "image_url": doc["image_url"],
                    "metadata": doc["metadata"]
                }
                results.append(result)

        return results


def demo_usage():
    """
    演示DashScope多模态RAG的使用方法
    """
    # 设置API密钥 (请替换为你的实际API密钥)
    API_KEY = os.environ.get("API_KEY_DASHSCOPE")

    # 创建RAG系统实例
    rag_system = DashScopeMultimodalRAG(API_KEY)

    # 添加示例文档
    print("=== 添加示例文档 ===")

    # 文档1: 技术文章 + 图片
    rag_system.add_document(
        doc_id="doc_001",
        text_content="Python是一种高级编程语言，以其简洁易读的语法而闻名。它广泛应用于Web开发、数据分析、人工智能等领域。",
        image_url="https://archive.biliimg.com/bfs/archive/87ab28155935de0788f38b1f2d69e26024bc39db.jpg",
        metadata={"title": "Python编程语言", "category": "编程语言"}
    )

    # 文档2: AI概念 + 图片
    rag_system.add_document(
        doc_id="doc_002",
        text_content="机器学习是人工智能的一个分支，通过算法使计算机能够从数据中学习并做出预测或决策。",
        image_url="https://pics2.baidu.com/feed/03087bf40ad162d977c8e5e7221a12e38b13cd78.jpeg@f_auto?token=1ae1e612f015f48e966983ca3eff255b",
        metadata={"title": "机器学习基础", "category": "人工智能"}
    )

    # 文档3: 深度学习 + 图片
    rag_system.add_document(
        doc_id="doc_003",
        text_content="深度学习是机器学习的一个子领域，使用多层神经网络来模拟人脑处理信息的方式。",
        image_url="https://i1.hdslb.com/bfs/archive/047e56f7097e487106c88c44728c3b95e1e0c990.jpg",
        metadata={"title": "深度学习概念", "category": "人工智能"}
    )

    # 构建索引
    print("\n=== 构建文档索引 ===")
    rag_system.build_index()

    # 执行搜索
    print("\n=== 执行多模态搜索 ===")

    # 示例1: 纯文本查询
    print("\n--- 示例1: 文本查询 'Python机器学习' ---")
    results = rag_system.search(query_text="Python机器学习", top_k=2)
    for i, result in enumerate(results, 1):
        print(f"{i}. 文档ID: {result['document_id']}")
        print(f"   相似度得分: {result['similarity_score']:.4f}")
        print(f"   内容预览: {result['text_content'][:50]}...")
        print(f"   图片URL: {result['image_url']}")
        print(f"   元数据: {result['metadata']}")

    # 示例2: 文本+图片查询
    print("\n--- 示例2: 文本+图片查询 ---")
    results = rag_system.search(
        query_text="深度学习",
        query_image_url="https://i1.hdslb.com/bfs/archive/047e56f7097e487106c88c44728c3b95e1e0c990.jpg",
        top_k=2
    )
    for i, result in enumerate(results, 1):
        print(f"{i}. 文档ID: {result['document_id']}")
        print(f"   相似度得分: {result['similarity_score']:.4f}")
        print(f"   内容预览: {result['text_content'][:50]}...")
        print(f"   图片URL: {result['image_url']}")
        print(f"   元数据: {result['metadata']}")


if __name__ == "__main__":
    demo_usage()
