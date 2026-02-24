import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

# Load dataset
data = pd.read_csv("analysis/swarm_ml_dataset.csv")

X = data[["N", "distance", "comm_range"]]
y = data["connectivity"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model R2 Score:", r2_score(y_test, y_pred))

# Save model
joblib.dump(model, "analysis/connectivity_model.pkl")

print("Model saved as connectivity_model.pkl")