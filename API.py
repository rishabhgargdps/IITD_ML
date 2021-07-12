#!/usr/bin/env python
# coding: utf-8

# In[31]:


import flask
import io
import string
import time
import os
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
import joblib as joblib
import json


# In[32]:

# In[33]:


def read_preprocess(file):
    df = pd.read_excel(file)
    pd.set_option('display.max_rows', 15)
    df = df.iloc[14:70, :]
    df.drop(df.columns[[0, 4, 7, 8, 9, 10, 11, 12]], axis=1, inplace = True)
    df.drop(df.loc[:, 'Unnamed: 14':'Unnamed: 15'].columns, axis = 1, inplace = True)
    df.rename(columns={'Unnamed: 13': 'RMS(g)'}, inplace = True)
    df.dropna(subset = ['RMS(g)'], inplace = True)
    return df

def read_preprocess2(file):
    df = pd.read_excel(file)
    pd.set_option('display.max_rows', 15)
    df = df.iloc[72:85, :]
    df.drop(df.columns[[0, 4, 8, 9, 10, 11, 12]], axis=1, inplace=True)
    df.drop(df.loc[:, 'Unnamed: 13':'Unnamed: 15'].columns, axis = 1, inplace = True)
    df.rename(columns={'RMS': 'Slope of the line'}, inplace=True)
    return df


# In[34]:


def predict(file, X_test):
    regressor = joblib.load(file) 
    y_predict = regressor.predict(X_test)
    return y_predict

def predict2(file, X_test):
    regressor2 = joblib.load(file)
    y_predict = regressor2.predict(X_test)
    return y_predict


# In[35]:


app = Flask(__name__)


# In[36]:


@app.route('/predict_rms', methods=['POST'])
def infer_image():
    # Catch the csv file from a POST request
    
    file_array = request.files.getlist("file[]")

    # Read and preprocess the csv file
    df = read_preprocess(file_array[0])

    # Return the prediction in JSON format
    X_test = df.iloc[:, 0:5]
    prediction = predict(file_array[1], X_test)
    lists = prediction.tolist()
    json_str = json.dumps(lists)
    return json_str

@app.route('/predict_slope', methods=['POST'])
def infer_image2():
    # Catch the csv file from a POST request
    
    file_array = request.files.getlist("file[]")

    # Read and preprocess the csv file
    df2 = read_preprocess2(file_array[0])

    # Return the prediction in JSON format
    X_test2 = df2.iloc[:, 0:5]
    prediction2 = predict2(file_array[1], X_test2)
    lists2 = prediction2.tolist()
    json_str2 = json.dumps(lists2)
    return json_str2
    

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


# In[37]:


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=os.environ.get('PORT', 5000))

