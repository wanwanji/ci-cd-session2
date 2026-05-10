import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score

# ==========================================
# 1. 加载数据
# ==========================================
df = pd.read_csv("./datasets/imdb_top_500.csv")
print(f"✅ 数据加载成功，共 {len(df)} 条影评")

# ==========================================
# 2. 加载 RoBERTa 模型
# ==========================================
print("🚀 正在加载 siebert/sentiment-roberta-large-english...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="siebert/sentiment-roberta-large-english",
    tokenizer="siebert/sentiment-roberta-large-english"
)
print("✅ 模型加载完成！")

# ==========================================
# 3. 批量预测
# ==========================================
texts = df["text"].tolist()
labels = df["label"].tolist()

preds = []
for text in texts:
    result = sentiment_pipeline(text[:512])[0]
    pred_label = 1 if result["label"] == "POSITIVE" else 0
    preds.append(pred_label)

# ==========================================
# 4. 准确率
# ==========================================
acc = accuracy_score(labels, preds)

print("-" * 40)
print(f"🎯 最终准确率: {acc:.2f}")
print("-" * 40)