import joblib
import numpy as np

model = joblib.load("analysis/connectivity_model.pkl")

N = 30
comm_range = 10

print(f"Finding optimal spacing for N={N}, comm_range={comm_range}")

for d in np.linspace(1, 6, 50):
    predicted = model.predict([[N, d, comm_range]])[0]

    if predicted >= 0.8:
        print(f"Optimal spacing ≈ {d:.2f}")
        break
else:
    print("No optimal spacing found.")