from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load('"C:\Users\sushant\Downloads\stacking_model.pkl"')  # Update this path

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from request
    data = request.get_json(force=True)
    # Assume 'data' is processed appropriately and is ready for prediction
    prediction = model.predict(data)

    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
