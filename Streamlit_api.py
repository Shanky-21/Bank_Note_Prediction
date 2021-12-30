# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 01:13:02 2021

@author: shash
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 22:11:54 2021

@author: shash
"""
#from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import streamlit as st

#from flasgger import Swagger

#app = Flask(__name__)
#Swagger(app)

pickle_in = open('classifier.pkl','rb')
classifier =pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "welcome All"

#@app.route('/predict',methods = ['GET'])
def predict_note_authentication(variance, skewness, curtosis, entropy):
    """Let's Authenticate the Banks Note
    This is using docstring for specifications.
    ---
    parameters:
        - name: variance
          in: query
          type: number
          required: true
        - name: skewness
          in: query
          type: number
          required: true
        - name: curtosis
          in: query
          type: number
          required: true
        - name: entropy
          in: query
          type: number
          required: entropy
    responses:
        200:
            description: The output values
          
    """
    #variance = request.args.get('variance')
    #skewness = request.args.get('skewness')
    #curtosis = request.args.get('curtosis')
    #entropy = request.args.get('entropy')
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    print(prediction)
    return "Hello the answer is: "+ str(prediction)

#@app.route('/predict_file', methods = ['POST'])
def predict_note_file():
    """Let's Authenticate the Banks Note
    This is using docstrings for specifications.
    ---
    parameters:
        - name: file
          in: formData
          type: file
          required: true
         
    responses:
        200:
            description: The output values
    
    """
    df_test = pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction = classifier.predict(df_test)
    
    return str(list(prediction))

def main():
    st.title("Bank Authenticator")
    html_temp = """
    <div style = "background-color:tomato;padding:10px">
    <h2 style = "color:white; text-align:center;">Streamlit Bank Authenticatior ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    variance = st.text_input("Variance", "Type Here")
    skewness = st.text_input("skewness","Type Here")
    curtosis = st.text_input("curtosis", "Type Here")
    entropy = st.text_input("entropy", "Type Here")
    result = ""
    if st.button("Predict"):
        result = predict_note_authentication(variance, skewness, curtosis, entropy)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Built with Streamlit")



if __name__ == '__main__':
    main()
