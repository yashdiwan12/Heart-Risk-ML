from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'model.pkl'

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    print(f"Warning: Model file not found at {MODEL_PATH}. Make sure to run Assignment-10.py first.")

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded on the server."}), 500
        
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Expected features in the correct order
        features_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                          'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        # Check if all required features are present
        missing_features = [feature for feature in features_order if feature not in data]
        if missing_features:
            return jsonify({"error": f"Missing features in JSON payload: {missing_features}"}), 400
            
        # Create a DataFrame from the JSON data to maintain feature names and order
        df = pd.DataFrame([data], columns=features_order)
        
        # Make a prediction
        prediction = model.predict(df)[0]
        
        # Format the output as requested by the assignment
        if prediction == 1:
            result = "Heart Disease Detected"
        else:
            result = "No Heart Disease Detected"
            
        return jsonify({"prediction": result})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Use 0.0.0.0 to bind to all interfaces (required for platforms like Render)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
