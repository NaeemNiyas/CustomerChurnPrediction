from flask import Flask, render_template, request, redirect, jsonify
import json
import numpy as np
import pickle
import requests
import xgboost as xgb
import re

app = Flask(__name__)
xgb_model = pickle.load(open('xgb_model.pkl', 'rb'))# loads ML model


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "GET":

        return render_template("index.html")
    
@app.route('/predict',methods=['POST'])# gets the values that were sent to '/predict' by 'index.html'
def predict():
        input_data = [float(x) for x in request.form.values()]# defines the form values in an array
        input_data_as_numpy_array= np.asarray(input_data)

        # reshape the numpy array as we are predicting for only on instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        #prediction = model.predict(input_data_reshaped)
        prediction = xgb_model.predict(input_data_reshaped) 
        output = prediction[0]# gets the prediction as a string
 
        print(output)
        if output == 1:
            res = "Yes, the customer is likely to leave"
        else:
            res = "No, the customer is unlikely to leave"
    
        return render_template("predict.html", res=res)


if __name__ == '__main__':
    app.run(debug=True)
