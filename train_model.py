import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = pd.read_csv('training_data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'linear_model.pkl')

with open('linear_model.txt', 'w') as f:
    f.write(f'Coefficients: {model.coef_}\nIntercept: {model.intercept_}\n')