# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import numpy as np
import pickle
import joblib
import streamlit as st


# Load saved model
loaded_model = pickle.load(open("C:/Users/kcmalema/Downloads/trained_model.sav", 'rb'))
loaded_vectorizer  = joblib.load("C:/Users/kcmalema/Downloads/vectorizer.pkl")


def topic_classifier(input_text):
    text_vector = loaded_vectorizer.transform([input_text])
    prediction = loaded_model.predict(text_vector)
    return prediction[0]


def main():
    
    st.title("Article Topic Classifier Web App")
    
    file_object = st.file_uploader("Upload article as text file")
    
    
    topic = ''
    
    if st.button('Classify'):
        input_data = file_object.read()
        topic = topic_classifier(input_data)
    
    st.success(topic)
    
    
if __name__ == "__main__":
    main()
