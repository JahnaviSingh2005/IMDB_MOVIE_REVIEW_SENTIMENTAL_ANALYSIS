import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
#load the word index
word_index = imdb.get_word_index()
reverse_word_index ={value:key for key,value in word_index.items()}
#load the pre-trained model with relu activation
model = load_model('simple_rnn_imdb.keras')
#model = load_model(r'C:\Users\singh\OneDrive\Documents\ann project\ann project\simple_rnn\simple_rnn_imdb.keras')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review
#step3 prediction function

#predictiom ftn
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment ='Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment,prediction[0][0]
#design streamlit
import streamlit as st

st.title('Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)

    # Make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    #display the result
    st.write(f'The sentiment of the review is {sentiment}')
    st.write(f'The prediction is {prediction[0][0]}')
else:
    st.write('No review was entered')


