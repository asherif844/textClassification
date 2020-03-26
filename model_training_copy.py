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

from azureml.core import Workspace, Datastore, Dataset
from azureml.core.run import Run
from azureml.core.model import Model

# subscription_id = '024d8ccf-00d6-468b-9028-67355d955e25'
# resource_group = 'Greenway'
# workspace_name = 'SOAP'
# workspace_region = 'eastus2'

# try:
#     ws = Workspace(subscription_id=subscription_id,
#                    resource_group=resource_group,
#                    workspace_name=workspace_name)
#     # write the details of the workspace to a configuration file to the notebook library
#     ws.write_config()
#     print("Workspace configuration succeeded. Skip the workspace creation steps below")
# except:
#     print("Workspace not accessible. A new workspace will be created now....")
#     ws = Workspace.create(name=workspace_name,
#                           subscription_id=subscription_id,
#                           resource_group=resource_group,
#                           location=workspace_region,
#                           create_resource_group=True,
#                           exist_ok=True)
#     ws.get_details()
#     ws.write_config()



run = Run.get_context()
# exp = run.experiment
# ws = run.experiment.workspace

print('Loading training data....')

datastore = ws.get_default_datastore()

traindata_paths = [(datastore, 'training_samples/training_samples.csv')]
lookup_paths = [(datastore, 'training_samples/lookup_table.csv')]
train_data = Dataset.Tabular.from_delimited_files(path=traindata_paths)
lookup_data = Dataset.Tabular.from_delimited_files(path=lookup_paths)
df = train_data.to_pandas_dataframe()
lookup = lookup_data.to_pandas_dataframe()

# df = pd.read_csv('data/training_samples.csv')
# lookup = pd.read_csv('data/lookup_table.csv')


print('Processing the training data....')
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

print('Exporting the training and testing sample shapes....')
print(train_X.shape, test_X.shape)

print('Training the model...')
run.log('test_size', 0.20)

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

run.log('accuracy', accuracy_score(predictions_NB, test_Y)*100)


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

model_file_name = 'naiveBayes.pkl'
model_path = os.path.join('./model', model_file_name)

print('Uploading the model into run artifacts...')
run.upload_file(name='./models/' + model_file_name, path_or_stream=model_path)

dirpath = os.getcwd()
print(dirpath)
print("Following files are uploaded: ")
print(run.get_file_names())

run.complete()
