import streamlit as st
import eda
import predict

st.sidebar.markdown('# About')
st.sidebar.write('''Click here to see Exploratory Data Analysis and Authenticity of Job Posting Predictions''')

navigation = st.sidebar.selectbox('Navigation',['Exploratory Data Analysis','Authenticity of Job Posting Predictions'])

if navigation == 'Exploratory Data Analysis':
    eda.run()
else:
    predict.run()