import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = pd.read_csv("../dataset/house_data.csv")
print("Dataset loaded")
print("Rows before cleaning:", data.shape[0])

# -----------------------------
# 2. Data Cleaning
# -----------------------------
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

# Remove outliers using IQR (Price)
Q1 = data["Price"].quantile(0.25)
Q3 = data["Price"].quantile(0.75)
IQR = Q3 - Q1
data = data[(data["Price"] >= Q1 - 1.5 * IQR) & (data["Price"] <= Q3 + 1.5 * IQR)]

print("Rows after cleaning:", data.shape[0])

# -----------------------------
# 3. Feature Engineering
# -----------------------------
# Encode Location
le = LabelEncoder()
data["Location"] = le.fit_transform(data["Location"])

# Log transform price (VERY IMPORTANT in real estate)
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
# 5. ML Pipeline (Scaler + Model)
# -----------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

pipeline.fit(X_train, y_train)

# -----------------------------
# 6. Evaluation
# -----------------------------
y_pred = pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))

print("R2 Score:", round(r2, 2))
print("MAE (₹):", round(mae, 2))
print("RMSE (₹):", round(rmse, 2))

# Cross Validation
cv_score = cross_val_score(pipeline, X, y, cv=5, scoring="r2")
print("CV R2 Avg:", round(cv_score.mean(), 2))

# -----------------------------
# 7. Save Model & Encoder
# -----------------------------
pickle.dump(pipeline, open("house_model.pkl", "wb"))
pickle.dump(le, open("location_encoder.pkl", "wb"))

print("✅ Real-world model & encoder saved successfully")
