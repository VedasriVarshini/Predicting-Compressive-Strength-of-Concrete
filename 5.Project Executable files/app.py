from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

model = pickle.load(open('cement.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index', methods=['POST'])
def index():
    return render_template('index1.html')

@app.route('/result', methods=['POST', 'GET'])
def prediction():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_aggregate', 'fine_aggregate ', 'age']
    x = pd.DataFrame(features_value, columns=features_name)

    x_scaled = scaler.transform(x)
    prediction = model.predict(x_scaled)
    
    return render_template('result2.html', prediction_text=f"Predicted Compressive Strength: {prediction[0]:.2f}")

if __name__ == "__main__":
    app.run(debug=True)
    app.run('0.0.0.0', 8088)
