from random import seed
from sre_parse import Tokenizer
from unittest.util import _MAX_LENGTH
import streamlit as st
from tensorflow.kera.preprocessing.text import tokenizer
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image

#import model
json_file = open("./model.json", "r")
loded_json_model = json_file.read()
json_file.close()

loaded_model = model_from_json(loded_json_model)

# load weights
loaded_model.load_weights("./model.h5")

with open("./bart-chalkboard-data.txt", 'r', encoding='utf-8') as file:
    data = file.read()


def generate_text(model, tokenizer, max_length, seed_text, n_words):
    text_generated = seed_text
    for i in range(n_words):
        encoded = tokenizer.texts_to_sequences([text_generated])[0]

        encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')

        # verbose set to 0 cuz dont need loading
        yhat = model.predict_classes(encoded, verbose=0)
        predicted_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                predicted_word = word
                break
        text_generated += ' ' + predicted_word
    return text_generated


tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

max_length = 14

st.title("teh Simpsons Chlakboard Gag text generator")

image = Image.open('./1.jpj')
st.image(image, use_column_width=True)

seed_text = st.text_input("Type a word or words you want to generate after")

if n_words and seed_text:
    st.header(generate_text(loaded_model, tokenizer,
              max_length-1, seed_text, n_words))
else:
    st.warning("please input word and a number")
