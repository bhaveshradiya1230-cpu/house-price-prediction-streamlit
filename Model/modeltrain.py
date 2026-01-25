import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. Load the dataset
data = pd.read_csv("../dataset/house_data.csv")
print("Dataset loaded")
print("Total rows before cleaning:", data.shape[0])

# 2. Data Cleaning

# Remove missing values
data.dropna(inplace=True)

# Remove duplicate rows
data.drop_duplicates(inplace=True)
print("Total rows after cleaning:", data.shape[0])

# 3. Feature Engineering

# Convert Location text into numbers
le = LabelEncoder()
data["Location"] = le.fit_transform(data["Location"])

# Select input features (X) and output (y)
X = data[["Area", "BHK", "Bathroom", "Location"]]
y = data["Price"]

# 4. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Evaluate the model
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", round(accuracy, 2))

# 7. Save the trained model and encoder
pickle.dump(model, open("house_model.pkl", "wb"))
pickle.dump(le, open("location_encoder.pkl", "wb"))

print("Model and encoder saved successfully")
