import pandas as pd
import numpy as np
import joblib

model = joblib.load("D:\\MLOPs_day1\\models\\model.pkl")

new_data = pd.DataFrame([[123, 56, 89]])

prediction = model.predict(new_data)

print("predicted sales: ", prediction)

