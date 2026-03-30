import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Sample dataset
data = {
    "study_hours": [2, 4, 6, 8, 3, 5, 7, 9],
    "sleep_hours": [6, 7, 8, 6, 5, 7, 8, 9],
    "break_time": [1, 1.5, 2, 2.5, 1, 1.5, 2, 2.5],
    "productivity": [4, 6, 7, 9, 3, 6, 8, 10]
}

df = pd.DataFrame(data)

X = df[["study_hours", "sleep_hours", "break_time"]]
y = df["productivity"]

model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
