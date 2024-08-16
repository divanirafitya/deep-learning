# import libraries
import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_hub as tf_hub
from tensorflow.keras.models import load_model

# import preprocessing
import re
import nltk
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# download the embedding layer
url = 'https://tfhub.dev/google/tf2-preview/nnlm-en-dim128-with-normalization/1'
hub_layer = tf_hub.KerasLayer(url, output_shape=[128], input_shape=[], dtype=tf.string)

# load model
model_nlp = load_model('improve_model_lstm.keras',custom_objects={'KerasLayer': hub_layer})

# text processing function
## create a function for text removal
def text_removal(text):
    '''
    Function to automate the deletion of unnecessary text.
    '''
    # convert text to lowercase
    text = text.lower()
    # hashtags removal
    text = re.sub(r"#\w+", " ", text)
    # newline removal (\n)
    text = re.sub(r"\n", " ", text)
    # URL removal
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)
    # symbol '&' removal
    text = re.sub(r"&amp;", " ", text) #in HTML
    # punctuation removal
    text = re.sub(r"[^\w\s]", " ", text)
    # non-letter removal (such as emoticon, symbol (like μ, $, 兀), etc
    text = re.sub(r"[^A-Za-z\s']", " ", text)
    # multiple spaces removal
    text = re.sub(r"\s+", " ", text)
    # whitespace removal
    text = text.strip()
    return text

## create a function for stopwords removal
def stopwords_removal(text):
    '''
    Function to automate the removal of stopwords ('and','or') and custom stopwords using the NLTK library.
    '''
    # defining stopwords
    stpwrd_eng = set(stopwords.words('english'))
    custom_stopwords = ['job','jobs','position','positions','career','careers']
    stpwrd_eng.update(custom_stopwords)
    # tokenization
    tokens = nltk.word_tokenize(text)
    # stopwords removal
    filtered_tokens = [word for word in tokens if word.lower() not in stpwrd_eng]
    # joining stopwords tokens back into a string
    cleaned_text = ' '.join(filtered_tokens)
    return cleaned_text

## create a function for lemmatization
def lemmatization(text):
    '''
    Function to soften text by returning a word to its base (lemmatization) using the NLTK library.
    '''
    # defining lemmatizer
    lemmatizer = WordNetLemmatizer()
    # tokenization
    tokens = nltk.word_tokenize(text)
    # lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # joining lemmatized tokens back into a string
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text

## combining tokens from preprocess
def text_preprocessing(text):
    '''
    Function to combine the results of text removal, stopwords removal, and lemmatization.
    '''
    text = text_removal(text)
    text = stopwords_removal(text)
    text = lemmatization(text)
    return text

def run():
    st.write('# Input Job Posting Information')

    # create form to input information
    with st.form(key='Diabetes Prediction'):
        
        # define all variables        
        st.write('## Company Profile Information')
        company_profile = st.text_input(label='Input Company Profile Text')
        st.markdown('---')

        st.write('## Job Description')
        description = st.text_input(label='Input Job Description Text')
        st.markdown('---')

        st.write('## Job Requirements')
        requirements = st.text_input(label='Input Job Requirements Text')
        st.markdown('---')

        # every form must have a submit button.
        submitted = st.form_submit_button('Predict')

    df_inf = {
        'company_profile': company_profile,
        'description': description,
        'requirements': requirements
    }

    df_inf = pd.DataFrame([df_inf])

    # combine column 'company_profile', 'description', and 'requirements'
    df_inf['text'] = df_inf['company_profile'] + ' ' + df_inf['description'] + ' ' + df_inf['requirements']
    # applying text preprocessing to the dataset
    df_inf['text_processed'] = df_inf['text'].apply(lambda x: text_preprocessing(x))
    st.dataframe(df_inf)

    # prediction
    if submitted:
        predict = model_nlp.predict(df_inf['text_processed'])
        predict = np.where(predict >= 0.5, 1, 0)
        for i in predict:
            if i == 0:
                st.write('## The job posting is real.')
            else:
                st.write('## The job posting is fake.')
        st.balloons()

if __name__ == '__main__':
   run()