import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# --- Synthetic weather dataset ---
np.random.seed(42)
n = 1000

temperature = np.random.uniform(-5, 45, n)
humidity    = np.random.uniform(20, 100, n)
wind_speed  = np.random.uniform(0, 80, n)
pressure    = np.random.uniform(960, 1040, n)
cloud_cover = np.random.uniform(0, 100, n)
visibility  = np.random.uniform(0, 20, n)

rain = (
    (humidity > 70).astype(int) * 2 +
    (cloud_cover > 60).astype(int) * 2 +
    (wind_speed > 25).astype(int) +
    (pressure < 1005).astype(int) +
    (visibility < 5).astype(int)
)
labels = (rain >= 4).astype(int)

df = pd.DataFrame({
    "temperature": temperature,
    "humidity":    humidity,
    "wind_speed":  wind_speed,
    "pressure":    pressure,
    "cloud_cover": cloud_cover,
    "visibility":  visibility,
    "rain":        labels
})

X = df.drop("rain", axis=1)
y = df["rain"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=["No Rain", "Rain"]))

with open("model (2).pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model saved: model (2).pkl")
