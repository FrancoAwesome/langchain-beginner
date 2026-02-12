import torchvision.models
from PIL import Image
import torch
from scipy.cluster.hierarchy import weighted
from torchvision.models import ResNet18_Weights
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from torchvision import transforms
import faiss

# 示例文本数据和图像数据
texts = ["这是一个关于猫的图片。", "这是一个关于狗的图片。"]
images = [Image.open("../../resources/images/cat.jpg"), Image.open("../../resources/images/dog.jpg")]  # 确保这些图片路径正确

# 图像转换为向量（这里简化处理，实际应用中可能需要更复杂的模型）
image_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
image_features = []
for img in images:
    img_tensor = image_transforms(img).unsqueeze(0)  # 添加批次维度
    # 这里使用预训练的模型提取特征，例如使用resnet18提取特征向量
    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT).eval()  # 确保模型处于评估模式
    with torch.no_grad():
        features = model(img_tensor).squeeze()  # 获取特征向量
    image_features.append(features.numpy())  # 转换为numpy数组以便于后续操作
image_features = np.stack(image_features)  # 堆叠成数组形式

# 使用faiss进行索引构建，这里以L2距离为例，适用于图像特征向量
index = faiss.IndexFlatL2(512)  # 假设每个图像的特征维度是512，根据实际情况调整
index.add(image_features)  # 添加图像特征到索引中

print(image_features.shape)
print(index.d)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")  # 这里用的是分类模型，实际应用中可以用生成模型如GPT-3等
text_input = "我需要一张猫的图片。"  # 用户输入的提示词
text_embeddings = model(**tokenizer(text_input, return_tensors="pt", padding=True)).last_hidden_state.mean(dim=1).detach().numpy()  # 获取文本嵌入并平均化处理以得到一个单一嵌入向量
distances, indices = index.search(text_embeddings, 1)  # 搜索最相似的图像特征向量索引

response = texts[indices[0][0]]  # 获取对应的文本描述作为响应
print(response)  # 输出响应结果，即找到的图像的描述文本。实际应用中可以进一步处理或展示图像等。