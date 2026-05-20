import json
import torch
import torch.nn as nn
import joblib

with open("model/config.json") as f:
    config = json.load(f)

vectorizer = joblib.load(
    "model/vectorizer.pkl"
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

model = SentimentNN(
    config["input_dim"]
)

model.load_state_dict(
    torch.load("model/model.pt")
)

model.eval()

text = "This movie was amazing"

vector = vectorizer.transform(
    [text]
).toarray()

tensor = torch.tensor(
    vector,
    dtype=torch.float32
)

with torch.no_grad():

    prediction = model(
        tensor
    ).item()

label = (
    "positive"
    if prediction >= 0.5
    else "negative"
)

print("Prediction:", label)