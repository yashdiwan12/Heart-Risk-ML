import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def main():
    # 1. Load the dataset using Pandas
    df = pd.read_csv('heart.csv')
    
    # 2. Print the first five records
    print("--- First Five Records ---")
    print(df.head())
    print("\n")
    
    # 3. Identify and document features and target
    print("--- Features and Target Variable ---")
    target = 'target'
    features = df.columns.tolist()
    features.remove(target)
    
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    print(f"Target Variable: {target}")
    print(f"Numerical Features: {numerical_features}")
    print(f"Categorical Features: {categorical_features}\n")
    
    # 4. Check for missing values
    print("--- Missing Values Check ---")
    missing_values = df.isnull().sum()
    print(missing_values)
    print("\n")
    
    # 5. Split dataset (80% training, 20% testing)
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data Split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.\n")
    
    # 6. Build classification model (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 7. Train and evaluate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("--- Model Performance ---")
    print(f"Accuracy Score: {accuracy:.4f}\n")
    
    # 8. Serialize and save the model
    joblib.dump(model, 'model.pkl')
    print("Model successfully saved as 'model.pkl'")

if __name__ == '__main__':
    main()
