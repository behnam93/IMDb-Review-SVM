import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pickle
data = pd.read_csv("/content/drive/MyDrive/Kaggle/imdb-reviews/IMDB Dataset.csv")
np.unique(data['sentiment'], return_counts=True)
le = LabelEncoder()
data['sentiment'] = le.fit_transform(data['sentiment'])
stop_words = list(STOP_WORDS)
X = data[['review']]
y = data['sentiment']
# lematize and stemmer
import nltk
stemmer = PorterStemmer()
lem = WordNetLemmatizer()
from nltk import word_tokenize
import nltk
import re
nltk.download('punkt')
nltk.download('wordnet')
dataset = pd.DataFrame(columns=['review'])
for index, row in X.iterrows():
    title_body_tokenized = word_tokenize(row['review'])
    title_body_tokenized_filtered = [w.lower() for w in title_body_tokenized if not w.lower() in stop_words]
    title_body_tokenized_filtered2 = [w for w in title_body_tokenized_filtered if len(w) > 2]
    title_body_tokenized_stemmed = [stemmer.stem(w) for w in title_body_tokenized_filtered2]
    title_body_tokenized_lematized = [lem.lemmatize(w) for w in title_body_tokenized_stemmed]
    s = re.sub('[^\w\s]', '', ' '.join(title_body_tokenized_lematized))
    s = re.sub("\d+", "", s)
    dataset.loc[index] = {
        'review': s,
    }
# add new column to dataset dataframe for labels
dataset['Label'] = y
vectorizer = TfidfVectorizer()
vectorizer.fit(dataset['review'])
X_fin = vectorizer.transform(dataset['review'])
import umap
reducer = umap.UMAP(n_components=1100,random_state=110)
X_reduced = reducer.fit_transform(X_fin)
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y,random_state=110)
from sklearn import svm
svmc = svm.SVC()
svmc.fit(X_train, y_train)
from sklearn.metrics import classification_report, confusion_matrix
y_pred = svmc.predict(X_test)
print(classification_report(y_test, y_pred))
