import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_and_prepare_data():
    """Load and prepare the US Accidents dataset"""
    print("Loading dataset...")
    df = pd.read_csv("US_Accidents_March23.csv")
    
    # Select relevant columns
    columns = ['Severity', 'Start_Lat', 'Start_Lng', 'Distance(mi)', 
              'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 
              'Wind_Speed(mph)', 'Precipitation(in)', 'Traffic_Signal', 
              'Junction', 'Railway', 'Nautical_Twilight']
    
    df = df[columns].copy()
    
    # Convert boolean columns to int
    df['Traffic_Signal'] = df['Traffic_Signal'].astype(int)
    df['Junction'] = df['Junction'].astype(int)
    df['Railway'] = df['Railway'].astype(int)
    
    # Convert Nautical_Twilight to numeric
    df['Nautical_Twilight'] = (
        df['Nautical_Twilight']
        .fillna('unknown')            
        .astype(str)
        .str.strip().str.lower()
        .map({'day': 1, 'night': 0, 'unknown': -1})
    )
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Prepare features and target
    X = df.drop('Severity', axis=1)
    y = df['Severity']
    
    return X, y

def train_model():
    """Train and save the model"""
    # Load and prepare data
    X, y = load_and_prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    print("\nModel Evaluation:")
    y_pred = rf_model.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and scaler
    print("\nSaving model and scaler...")
    joblib.dump(rf_model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Model and scaler saved successfully!")

if __name__ == '__main__':
    train_model() 