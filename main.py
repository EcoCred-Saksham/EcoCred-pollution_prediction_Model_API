from flask import Flask, request, jsonify
from joblib import load
import numpy as np
from datetime import datetime

app = Flask(__name__)

def receive_wallet_address():
    data = request.get_json()
    if not data or 'val1' not in data or 'val2' not in data:
        return jsonify({"error": "Invalid input"}), 400

    val1 = int(data['val1'])
    val2 = int(data['val2'])

    # Load the trained model
    model = load('model.pkl')

    # Get the current year
    curr_yr = datetime.now().year

    res = []
    res.append(model.predict(np.array([[val1, val2]])))
    res.append(model.predict(np.array([[res[-1][0], val2]])))
    res.append(model.predict(np.array([[res[-2][0], res[-1][0]]])))

    predictions = [abs(float(r))%400 for r in res]

    return jsonify({"prediction": predictions}), 200

@app.route('/predict', methods=['POST'])
def predict():
    return receive_wallet_address()

if __name__ == '__main__':
    app.run(debug=True)
