import json
import os
import pickle
import re
from collections import defaultdict

import nltk
import numpy as np
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import model_selection, naive_bayes, svm
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_csv('data/training_samples.csv')
lookup = pd.read_csv('data/lookup_table.csv')

df = df[['0', '1']]
df.columns = ['text', 'label']

df2 = pd.merge(df, lookup, left_on='label', right_on='index', how='outer')

df2['Area'].value_counts()
df2['Area'] = df2['Area'].str.upper()
df2 = df2[['text', 'Area']]

df2 = df2[~df2['Area'].isin([' ', 'DELETE', 'MEDICATION', 'ALLERGIES', ''])]

df2['text'] = df2['text'].apply(lambda x: re.sub('[^a-zA-Z0-9 \n\.]', '', x))


df2 = df2[~df2['text'].str.startswith('PLAN')]
df2 = df2[~df2['text'].apply(lambda x: len(x) <= 5)]
df2 = df2.drop_duplicates()
df2 = df2.dropna(axis=0)


def text_analysis(raw_data):
    try:
        i = raw_data
        analyser = SentimentIntensityAnalyzer()
        subobj = {'sentiment': analyser.polarity_scores(i), 'subjectivity': round(TextBlob(
            i).sentiment.subjectivity, 4), 'objectivity': round((1 - TextBlob(i).sentiment.subjectivity), 4)}
    except Exception as e:
        return e
    return str(subobj)

# sample_scenarios = ['I like this', 'I love this', 'I hate this']
# for i in sample_scenarios:
#   print(text_analysis(i))


analyser = SentimentIntensityAnalyzer()
df2['negative'] = df2['text'].apply(
    lambda x: analyser.polarity_scores(x).get('neg'))
df2['neutral'] = df2['text'].apply(
    lambda x: analyser.polarity_scores(x).get('neu'))
df2['positive'] = df2['text'].apply(
    lambda x: analyser.polarity_scores(x).get('pos'))
df2['compound'] = df2['text'].apply(
    lambda x: analyser.polarity_scores(x).get('compound'))

df2['subjective'] = df2['text'].apply(
    lambda x: round(TextBlob(x).sentiment.subjectivity, 4))
df2['objective'] = df2['text'].apply(
    lambda x: round(1 - TextBlob(x).sentiment.subjectivity, 4))

df2['label'] = df2['Area']
# df2['label'][(df2['Area'] == 'SUBJECTIVE') & (
#     df2['objective'] >= 0.50)] = 'OBJECTIVE'


train_X, test_X, train_Y, test_Y = model_selection.train_test_split(
    df2['text'], df2['label'], test_size=0.2)

print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)

Encoder = LabelEncoder()
labels = train_Y
train_Y = Encoder.fit_transform(train_Y)
test_Y = Encoder.fit_transform(test_Y)

lookup = pd.DataFrame({'labelName': labels.values, 'LabelCode': train_Y})
lookup = lookup.drop_duplicates()

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(df['text'])

Train_X_Tfidf = Tfidf_vect.transform(train_X)
Test_X_Tfidf = Tfidf_vect.transform(test_X)

# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf, train_Y)

# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",
      accuracy_score(predictions_NB, test_Y)*100)


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
        prediction_output = {f'Sentence {i[0]+1}': label, 'Assessment': predictions_temp[0],
                             'Objective': predictions_temp[1], 'Plan': predictions_temp[2], 'Subjective': predictions_temp[3]}
        text_predictions.append(prediction_output)
    return str(text_predictions)


nb_model_location = os.path.join(os.getcwd(), 'models')
os.makedirs(nb_model_location, exist_ok=True)

joblib.dump(Naive, filename=os.path.join(nb_model_location, 'naiveBayes.pkl'))
joblib.dump(Tfidf_vect, filename=os.path.join(nb_model_location, 'vect.pkl'))
joblib.dump(lookup, filename=os.path.join(nb_model_location, 'lookup.pkl'))


sample = 'This is awesome. This is horrible'
model_run(sample)
