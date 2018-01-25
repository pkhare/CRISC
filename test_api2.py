# -*- coding: utf-8 -*-
import os
import sys
from json import dumps
import json
import requests
import arff
import pandas as pd
import numpy as np
import scipy as sp
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import string
from nltk.tokenize import regexp_tokenize
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.externals import joblib
import pickle

reload(sys)
sys.setdefaultencoding('utf8')

stemmer = PorterStemmer()


def tokenize_and_stem(text):
    # tokens = word_tokenize(text)
    tokens = regexp_tokenize(text, pattern=r"\s|[\.,:;'()?!]", gaps=True)
    # strip out punctuation and make lowercase
    tokens = [token.lower().strip(string.punctuation)
              for token in tokens if token.isalnum()]

    # now stem the tokens
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens

################################


filename = "/Users/pk4634/crisc/SVC_classifier_model.pkl"
loaded_model = pickle.load(open(filename, "rb"))

filename_vectorize = "/Users/pk4634/crisc/vectorizer.pkl"
filename_tf_transform = "/Users/pk4634/crisc/tf_transform.pkl"

vectorizer = pickle.load(open(filename_vectorize, "rb"))
tf_transform = pickle.load(open(filename_tf_transform, "rb"))
################################


def call_other_api(tweet_str):

    tweet = tweet_str
    payload = {'tweet': tweet}

    api_response = requests.get('http://localhost:4567/crisc', params=payload)
    if(api_response.ok):
        json_reponse = json.loads(api_response.content)
        '''for key in json_reponse:
                print key + " : " + str(json_reponse[key])'''

        enr_tweet = json_reponse["enr_tweet"]
        Token_count = json_reponse["Token_count"]
        TweetL = json_reponse["TweetL"]
        Verbs = json_reponse["Verbs"]
        Nouns = json_reponse["Nouns"]
        HashTag = json_reponse["HashTag"]
        Pronouns = json_reponse["Pronouns"]
        Readability = json_reponse["Readability"]

        data_test = np.array([[enr_tweet, Nouns, Verbs, Pronouns,
                               TweetL, Token_count, HashTag, Readability]])

        return data_test


text_str = "RT @PennStateIFC: With the current devastation in Houston, we are pledging $0.15 for every RT this gets! Please forward this along to…"
if text_str.strip() == '':
    print text_str.strip() + '\n'
    print 'null'
else:
    data_test = call_other_api(text_str)

    X_test = data_test[:, 0:8]

    t = ['Document', 'NumberOfNouns', 'NumberOfVerbs', 'NumberOfPronouns',
         'TweetLength', 'NumberOfWords', 'NumberOfHashTag', 'Readability']

    frm_test = pd.DataFrame(X_test, columns=t)

    test_vectorize = vectorizer.transform(frm_test.Document)
    test_tf_vectorize = tf_transform.transform(test_vectorize)

    X_data_test = sp.sparse.hstack((test_tf_vectorize, frm_test[['NumberOfNouns', 'NumberOfVerbs', 'NumberOfPronouns',
                                                                 'TweetLength', 'NumberOfWords', 'NumberOfHashTag', 'Readability']].values.astype(np.float32)), format='csr')

    ################################
    Y_test_predict = loaded_model.predict(X_data_test)

    print Y_test_predict[0]

    if Y_test_predict == 1.0:
        print 'related'
    if Y_test_predict == 0.0:
        print 'non-related'
