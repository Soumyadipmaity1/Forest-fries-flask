import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from preprocessing import preprocess_data

# Load dataset
df = pd.read_csv("forestfires.csv")

# Preprocess data
df_processed = preprocess_data(df, training=True)

# Split into features (X) and target (y)
X = df_processed.drop(columns=["area"])  # Assuming 'area' is the target variable
y = df_processed["area"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model training complete. 'model.pkl' saved successfully!")
