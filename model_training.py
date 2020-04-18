# import python core libraries
import json
import os
import pickle
import re
from collections import defaultdict

# import external python libraries

import nltk
import numpy as np
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import model_selection, naive_bayes, svm, utils
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


print('Loading the Training Data....')
df = pd.read_csv('data/spam_encoded.csv', encoding="utf-8")

df.head()

df = df[df.columns[:2]]

df.columns = ['label', 'text']

df['label'].value_counts()

print('Begin the data munging phase...')


def text_analysis(raw_data):
    try:
        i = raw_data
        analyser = SentimentIntensityAnalyzer()
        subobj = {'sentiment': analyser.polarity_scores(i), 'subjectivity': round(TextBlob(
            i).sentiment.subjectivity, 4), 'objectivity': round((1 - TextBlob(i).sentiment.subjectivity), 4)}
    except Exception as e:
        return e
    return str(subobj)


analyser = SentimentIntensityAnalyzer()

df['negative'] = df['text'].apply(
    lambda x: analyser.polarity_scores(x).get('neg'))
df['neutral'] = df['text'].apply(
    lambda x: analyser.polarity_scores(x).get('neu'))
df['positive'] = df['text'].apply(
    lambda x: analyser.polarity_scores(x).get('pos'))
df['compound'] = df['text'].apply(
    lambda x: analyser.polarity_scores(x).get('compound'))
df['subjective'] = df['text'].apply(
    lambda x: round(TextBlob(x).sentiment.subjectivity, 4))
df['objective'] = df['text'].apply(lambda x: round(
    1 - TextBlob(x).sentiment.subjectivity, 4))

print('we have a data imbalance....')
print('we will address the imbalance in the training split step....')

df_spam = df[df['label'] == 1]
df_ham = df[df['label'] == 0]

df_ham_downsamples = utils.resample(
    df_ham, replace=False, n_samples=len(df_spam), random_state=12345)

df2 = pd.concat([df_spam, df_ham_downsamples])

df2 = df2.sample(frac=1)

print('Begin the training split step....')
train_X, test_X, train_Y, test_Y = model_selection.train_test_split(
    df2['text'], df2['label'], test_size=0.2)

print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)

Encoder = LabelEncoder()
train_Y = Encoder.fit_transform(train_Y)
test_Y = Encoder.fit_transform(test_Y)

lookup = pd.DataFrame({'labelName': ['not spam', 'spam'], 'LabelCode': [0, 1]})


Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(df['text'])

Train_X_Tfidf = Tfidf_vect.transform(train_X)
Test_X_Tfidf = Tfidf_vect.transform(test_X)

print('Begin the training proces....')
# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf, train_Y)

# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",
      accuracy_score(predictions_NB, test_Y)*100)


print('Split text into potential sentences....')

"""
This code for the function split_into_sentences() was obtained from user (D Greenberg) off of stackoverflow.com
The link to this code can be obtained from the following link:
https://stackoverflow.com/a/31505798/6594800

"""


alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"


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


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


def add_an_ending(text):
    if text[-1] not in ['!', ',', '.', '\n', '?']:
        text += '.'
    return text


def model_run(text):
    text_predictions = []
    text = add_an_ending(text)
    for i in enumerate(split_into_sentences(text)):
        label = str(i[1])
        df_temp = pd.DataFrame([label], columns=['text'])
        df_temp['text'] = [entry.lower() for entry in df_temp['text']]
        df_temp['text'] = [word_tokenize(entry) for entry in df_temp['text']]
        df_temp['text_final'] = [" ".join(review)
                                 for review in df_temp['text'].values]
        Tfidf = Tfidf_vect.transform(df_temp['text_final'])
        predictions_temp = Naive.predict_proba(Tfidf)[0]
        prediction_output = {f'Sentence {i[0]+1}': label, 'Not Spam': predictions_temp[0],
                             'Spam': predictions_temp[1]}
        text_predictions.append(prediction_output)
    return str(text_predictions)


nb_model_location = os.path.join(os.getcwd(), 'models')
os.makedirs(nb_model_location, exist_ok=True)

print('model artifact exporting....')
joblib.dump(Naive, filename=os.path.join(nb_model_location, 'naiveBayes.pkl'))
joblib.dump(Tfidf_vect, filename=os.path.join(
    nb_model_location, 'vectorization.pkl'))
joblib.dump(lookup, filename=os.path.join(nb_model_location, 'lookup.pkl'))


sample = 'This is awesome. This is horrible'
print(model_run(sample))
