import json
import os
import pickle
import re

import nltk
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, flash
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.externals import joblib

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"


# retrieve the path to the model file using the model name
models_folder = os.path.join(os.getcwd(), 'models')
model = joblib.load(os.path.join(models_folder, 'naiveBayes.pkl'))
Tfidf_vect = joblib.load(os.path.join(models_folder, 'vectorization.pkl'))
lookup = joblib.load(os.path.join(models_folder, 'lookup.pkl'))


def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms+" "+starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" +
                  alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets +
                  "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" "+suffixes+"[.] "+starters, " \\1<stop> \\2", text)
    text = re.sub(" "+suffixes+"[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if "\"" in text:
        text = text.replace(".\"", "\".")
    if "!" in text:
        text = text.replace("!\"", "\"!")
    if "?" in text:
        text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def add_an_ending(text):
    if text[-1] not in ['!', ',', '.', '\n', '?']:
        text += '.'
    return text


app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        text_predictions= []
        text = add_an_ending(request.form.get('eMail'))
        for i in enumerate(split_into_sentences(text)):
            label = str(i[1])
            df_temp = pd.DataFrame([label], columns=['text'])
            df_temp['text'] = [entry.lower() for entry in df_temp['text']]
            df_temp['text'] = [word_tokenize(entry) for entry in df_temp['text']]
            df_temp['text_final'] = [" ".join(review)
                                    for review in df_temp['text'].values]
            Tfidf = Tfidf_vect.transform(df_temp['text_final'])
            predictions_temp = model.predict_proba(Tfidf)[0]
            prediction_output = {
                f'Sentence {i[0]+1}': label, 'Not Spam': predictions_temp[0], 'Spam': predictions_temp[1]}
            text_predictions.append(prediction_output)
        return """
                  <p>The ham/spam breakout for this eMail is: {}</p>
                  
                  """.format(text_predictions)

    return '''<form method="POST">
                  eMail: <input type="text" name="eMail"><br>
                  <input type="submit" value="Submit"><br>
              </form>'''


@app.route('/soap', methods=['POST'])
def run():
    text_predictions = []
    text2 = request.json
    text = add_an_ending(text2.get('text'))
    for i in enumerate(split_into_sentences(text)):
        label = str(i[1])
        df_temp = pd.DataFrame([label], columns=['text'])
        df_temp['text'] = [entry.lower() for entry in df_temp['text']]
        df_temp['text'] = [word_tokenize(entry) for entry in df_temp['text']]
        df_temp['text_final'] = [" ".join(review)
                                 for review in df_temp['text'].values]
        Tfidf = Tfidf_vect.transform(df_temp['text_final'])
        predictions_temp = model.predict_proba(Tfidf)[0]
        prediction_output = {
            f'Sentence {i[0]+1}': label, 'Not Spam': predictions_temp[0], 'Spam': predictions_temp[1]}
        text_predictions.append(prediction_output)
    return jsonify(text_predictions)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
