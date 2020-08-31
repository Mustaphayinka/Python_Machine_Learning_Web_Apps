import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, url_for, request
model = pickle.load(open('model1.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    arr = [data1, data2, data3, data4]
    clean_data = [float(i) for i in arr]
    ex1 = np.array(clean_data).reshape(1, -1)
    pred = model.predict(ex1)

    return render_template('index.html', prediction = pred)


if __name__ == "__main__":
    app.run(debug = True)