import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = pd.read_csv("../Dataset/house_data.csv")
print("Dataset Loaded:", data.shape)

# -----------------------------
# 2. Fast Cleaning (NO heavy ops)
# -----------------------------
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

# Remove extreme price outliers (fast method)
data = data[data["Price"] < data["Price"].quantile(0.99)]

# -----------------------------
# 3. Feature Engineering
# -----------------------------
le = LabelEncoder()
data["Location"] = le.fit_transform(data["Location"])

# Log target (boosts accuracy)
data["Price"] = np.log1p(data["Price"])

X = data[["Area", "BHK", "Bathroom", "Location"]]
y = data["Price"]

# -----------------------------
# 4. Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. FAST HYBRID MODELS
# -----------------------------
lr = LinearRegression()

rf = RandomForestRegressor(
    n_estimators=120,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

et = ExtraTreesRegressor(
    n_estimators=150,
    max_depth=14,
    random_state=42,
    n_jobs=-1
)

# -----------------------------
# 6. VOTING HYBRID (FAST)
# -----------------------------
hybrid_model = VotingRegressor(
    estimators=[
        ("lr", lr),
        ("rf", rf),
        ("et", et)
    ],
    weights=[1, 2, 3]  # ExtraTrees priority
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", hybrid_model)
])

# -----------------------------
# 7. Train (FAST ⚡)
# -----------------------------
pipeline.fit(X_train, y_train)

# -----------------------------
# 8. Evaluation
# -----------------------------
y_pred = pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))

print("\n⚡ FAST HYBRID MODEL RESULTS ⚡")
print("R2 Score:", round(r2, 3))
print("MAE (₹):", round(mae, 2))
print("RMSE (₹):", round(rmse, 2))

# Cross-Validation (FAST CV)
cv = cross_val_score(pipeline, X, y, cv=3, scoring="r2")
print("CV R2 Avg:", round(cv.mean(), 3))

# -----------------------------
# 9. Save Model
# -----------------------------
pickle.dump(pipeline, open("house_model.pkl", "wb"))
pickle.dump(le, open("location_encoder.pkl", "wb"))

print("\n✅ FAST Hybrid Model Saved Successfully")
