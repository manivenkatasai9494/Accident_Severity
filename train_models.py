import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic data for training"""
    np.random.seed(42)
    
    # Generate features
    data = {
        'start_lat': np.random.uniform(47.0, 48.0, n_samples),
        'start_lng': np.random.uniform(-123.0, -122.0, n_samples),
        'distance': np.random.uniform(0.5, 5.0, n_samples),
        'temperature': np.random.uniform(20, 90, n_samples),
        'humidity': np.random.uniform(30, 100, n_samples),
        'visibility': np.random.uniform(0, 20, n_samples),
        'wind_speed': np.random.uniform(0, 30, n_samples),
        'precipitation': np.random.uniform(0, 2, n_samples),
        'traffic_signal': np.random.randint(0, 2, n_samples),
        'junction': np.random.randint(0, 2, n_samples),
        'railway': np.random.randint(0, 2, n_samples),
        'nautical_twilight': np.random.randint(0, 2, n_samples)
    }
    
    # Generate target (severity 1-4)
    # Create more complex patterns for severity
    severity = np.zeros(n_samples)
    for i in range(n_samples):
        # Base risk factors
        risk = 0
        
        # Weather factors
        if data['visibility'][i] < 5:
            risk += 1
        if data['wind_speed'][i] > 20:
            risk += 1
        if data['precipitation'][i] > 0.1:
            risk += 1
        if data['temperature'][i] < 32:
            risk += 1
            
        # Traffic factors
        if data['traffic_signal'][i] == 1:
            risk += 1
        if data['junction'][i] == 1:
            risk += 1
        if data['railway'][i] == 1:
            risk += 1
            
        # Time factors
        if data['nautical_twilight'][i] == 1:
            risk += 1
            
        # Distance factor
        if data['distance'][i] < 1.0:
            risk += 1
            
        # Add some randomness
        risk += np.random.randint(-1, 2)
        
        # Convert risk to severity (1-4)
        severity[i] = max(1, min(4, risk))
    
    data['severity'] = severity
    return pd.DataFrame(data)

def train_models():
    """Train and save both models"""
    # Generate synthetic data
    print("Generating synthetic data...")
    df = generate_synthetic_data(10000)
    
    # Prepare features and target
    X = df.drop('severity', axis=1)
    y = df['severity']
    
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
    
    # Train Gradient Boosting
    print("Training Gradient Boosting model...")
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    
    # Evaluate models
    print("\nModel Evaluation:")
    print("\nRandom Forest:")
    rf_pred = rf_model.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, rf_pred))
    
    print("\nGradient Boosting:")
    gb_pred = gb_model.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, gb_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, gb_pred))
    
    # Save models and scaler
    print("\nSaving models and scaler...")
    joblib.dump(rf_model, 'rf_model.joblib')
    joblib.dump(gb_model, 'gb_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Models and scaler saved successfully!")

if __name__ == '__main__':
    train_models() 