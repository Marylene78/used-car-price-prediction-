from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

# Load the trained model
model = joblib.load("xgboost_used_car_price_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)

        # Debugging
        print(f"Received features: {features.shape}, Model expects: {model.n_features_in_}")

        if features.shape[1] != model.n_features_in_:
            return jsonify({"error": f"Feature shape mismatch. Expected: {model.n_features_in_}, got: {features.shape[1]}"})

        prediction = model.predict(features)

        return jsonify({"predicted_price": float(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
