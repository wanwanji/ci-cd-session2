import numpy as np
import pandas as pd
import json
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================
# 1. 加载数据
# ==========================================
try:
    df = pd.read_csv("./datasets/imdb_top_500.csv")
    with open("./datasets/tiny_glove.json", "r") as f:
        glove = json.load(f)
    
    print("✅ 数据加载成功")
    print(f"数据集大小: {len(df)} 条")
    print(f"词向量库大小: {len(glove)} 个词")
except FileNotFoundError:
    print("❌ 错误：找不到数据集文件。请检查 './datasets/' 文件夹是否存在以及文件路径是否正确。")
    exit()

# ==========================================
# 2. 文本清洗与分词
# ==========================================
def tokenize(text):
    """
    将文本转换为小写，去除标点符号和数字，然后分词
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()

# ==========================================
# 3. 将单条评论转换为向量
# ==========================================
def get_embedding(text, glove, dim=50):
    """
    将一句话转换为一个向量（取所有词向量的平均值）
    """
    tokens = tokenize(text)
    
    vectors = [
        np.array(glove[word])
        for word in tokens
        if word in glove
    ]
    
    if len(vectors) == 0:
        return np.zeros(dim)
    
    return np.mean(vectors, axis=0)

# ==========================================
# 4. 构建特征矩阵 (X) 和 标签 (y)
# ==========================================
print("\n🔄 正在将所有评论转换为向量，请稍候...")
X = np.array([get_embedding(text, glove) for text in df["text"]])
y = df["label"].values
texts = df["text"].values

print(f"特征矩阵形状: {X.shape}")
print(f"标签形状: {y.shape}")

# ==========================================
# 5. 划分训练集和测试集
# ==========================================
X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
    X,
    y,
    texts,
    test_size=0.2,
    random_state=42
)

print(f"训练集样本数: {len(X_train)}")
print(f"测试集样本数: {len(X_test)}")

# ==========================================
# 6. 特征标准化
# ==========================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("✅ 特征标准化完成")

# ==========================================
# 7. 训练逻辑回归模型
# ==========================================
model = LogisticRegression()
model.fit(X_train, y_train)
print("✅ 模型训练完成")

# ==========================================
# 8. 评估模型
# ==========================================
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print("-" * 30)
print(f"📈 训练集准确率: {train_acc:.2f}")
print(f"📉 测试集准确率: {test_acc:.2f}")
print("-" * 30)

# ==========================================
# 9. 查看具体预测结果
# ==========================================
print("\n🔍 随机抽取 3 条测试集评论进行预测展示：")
import random
indices = random.sample(range(len(X_test)), 3)

for i in indices:
    print(f"\n评论内容: {text_test[i][:100]}...")
    print(f"真实标签: {y_test[i]}")
    print(f"预测标签: {test_pred[i]}")
    if y_test[i] == test_pred[i]:
        print("结果: ✅ 正确")
    else:
        print("结果: ❌ 错误")

# ==========================================
# 10. 预测新评论
# ==========================================
print("\n🚀 尝试预测全新的评论：")
sample_reviews = [
    "This movie was fantastic with brilliant acting",
    "I hated this movie it was boring and terrible",
    "The film was okay not great but not bad"
]

sample_X = np.array([get_embedding(text, glove) for text in sample_reviews])
sample_X = scaler.transform(sample_X)

sample_preds = model.predict(sample_X)

for review, pred in zip(sample_reviews, sample_preds):
    sentiment = "😊 正面 (Positive)" if pred == 1 else "😞 负面 (Negative)"
    print(f"评论: {review}")
    print(f"预测结果: {sentiment}")
    print("-" * 20)