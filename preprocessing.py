import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import numpy as np

def preprocess_data():
    # Load the dataset
    df = pd.read_csv('forestfires.csv')
    
    # Convert 'area' to log scale to handle skewness
    df['area'] = np.log1p(df['area'])
    
    # Encode categorical variables
    le_month = LabelEncoder()
    le_day = LabelEncoder()
    df['month'] = le_month.fit_transform(df['month'])
    df['day'] = le_day.fit_transform(df['day'])
    
    # Feature selection - removing X,Y coordinates as they may not be meaningful predictors
    X = df.drop(['area', 'X', 'Y'], axis=1)
    y = df['area']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=200, 
                                max_depth=15, 
                                min_samples_split=5, 
                                random_state=42,
                                n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, train_pred))}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, test_pred))}")
    
    # Save artifacts
    joblib.dump(model, 'models/random_forest_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(le_month, 'models/month_encoder.joblib')
    joblib.dump(le_day, 'models/day_encoder.joblib')
    
    return model, scaler, le_month, le_day

def predict_fire_risk(input_data):
    # Load artifacts
    model = joblib.load('models/random_forest_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    le_month = joblib.load('models/month_encoder.joblib')
    le_day = joblib.load('models/day_encoder.joblib')
    
    # Create DataFrame from input
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical variables
    input_df['month'] = le_month.transform([input_data['month']])[0]
    input_df['day'] = le_day.transform([input_data['day']])[0]
    
    # Remove X,Y if present
    if 'X' in input_df.columns:
        input_df.drop(['X', 'Y'], axis=1, inplace=True)
    
    # Scale features
    scaled_input = scaler.transform(input_df)
    
    # Predict
    log_area = model.predict(scaled_input)[0]
    area = np.expm1(log_area)
    
    # Determine risk level
    if area < 1:
        risk = "Low"
    elif 1 <= area < 10:
        risk = "Moderate"
    elif 10 <= area < 100:
        risk = "High"
    else:
        risk = "Very High"
    
    return area, risk