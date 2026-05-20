import os
import json
import joblib
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

os.makedirs("model", exist_ok=True)

df = pd.read_csv("data/imdb_balanced_10k.csv")

df["label"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

X_train, X_test, y_train, y_test = train_test_split(
    df["review"],
    df["label"],
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train).toarray()

X_test_vec = vectorizer.transform(X_test).toarray()

input_dim = X_train_vec.shape[1]

X_train_tensor = torch.tensor(
    X_train_vec,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train.values,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test_vec,
    dtype=torch.float32
)

class SentimentNN(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.fc1 = nn.Linear(input_dim, 128)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        x = self.sigmoid(x)

        return x

model = SentimentNN(input_dim)

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 5

for epoch in range(epochs):

    optimizer.zero_grad()

    outputs = model(
        X_train_tensor
    ).squeeze()

    loss = criterion(
        outputs,
        y_train_tensor
    )

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Loss: {loss.item():.4f}")

with torch.no_grad():

    predictions = model(
        X_test_tensor
    ).squeeze()

    predictions = (
        predictions >= 0.5
    ).int().numpy()

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"Accuracy: {accuracy}")

torch.save(
    model.state_dict(),
    "model/model.pt"
)

joblib.dump(
    vectorizer,
    "model/vectorizer.pkl"
)

config = {
    "input_dim": input_dim
}

with open(
    "model/config.json",
    "w"
) as f:

    json.dump(config, f)

metrics = {
    "accuracy": float(accuracy)
}

with open(
    "model/metrics.json",
    "w"
) as f:

    json.dump(metrics, f)

print("Training completed successfully.")