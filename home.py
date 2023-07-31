import streamlit as st

from modules.util import predict

st.header("Detection Hate Speech")

st.write('''
         Detection Hate Speech using using Deep Learning Algorithms - Convolutional Neural Network (CNN). 
         ''')

text = st.text_area("Enter your text here: ",height=200,help='English Text is recommended !')

if st.button('predict'):
    result = predict(text)
    st.subheader("Result")
    st.write(result)