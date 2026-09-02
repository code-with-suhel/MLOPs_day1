import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import joblib


# Load Data
df = pd.read_csv("D:\\MLOPs_day1\\data\\data.csv")

# train_test_split

X,y = df.drop(columns = ["Sales"]), df["Sales"]
Xtrain, xtest, ytrain, ytest = train_test_split(X,y, test_size= 0.2, random_state=42)


# Linear Regression Model

model = LinearRegression()
model.fit(Xtrain, ytrain)

# Model dump

joblib.dump(model, "models/model.pkl")




