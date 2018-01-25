# -*- coding: utf-8 -*-
import os
import sys
import xgboost
from flask import Flask
from flask import request
from flask import jsonify
from flask_restful import Resource, Api
from flask_restful import reqparse
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

app = Flask(__name__)
#api = Api(app)

################################

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


filename = "SVC_classifier_model_9events.pkl"
loaded_model = pickle.load(open(filename, "rb"))

filename_vectorize = "vectorizer_9events.pkl"
filename_tf_transform = "tf_transform_9events.pkl"

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

    # def get(self, file_str):


@app.route('/crisc', methods=['GET'])
def get_twt():

    file_str = request.args.get('twstr', type=str)

    ################################
    data_test = call_other_api(file_str)
    #data_test = data_test_json['data_test']
    X_test = data_test[:, 0:8]

    ################################

    t = ['Document', 'NumberOfNouns', 'NumberOfVerbs', 'NumberOfPronouns',
         'TweetLength', 'NumberOfWords', 'NumberOfHashTag', 'Readability']

    frm_test = pd.DataFrame(X_test, columns=t)

    test_vectorize = vectorizer.transform(frm_test.Document)
    test_tf_vectorize = tf_transform.transform(test_vectorize)

    X_data_test = sp.sparse.hstack((test_tf_vectorize, frm_test[['NumberOfNouns', 'NumberOfVerbs', 'NumberOfPronouns',
                                                                 'TweetLength', 'NumberOfWords', 'NumberOfHashTag', 'Readability']].values.astype(np.float32)), format='csr')

    ################################
    Y_test_predict = loaded_model.predict(X_data_test)
    # print (metrics.f1_score(Y_test, Y_test_predict))

    # return {'departments': 'yess'}
    # return {'departments': metrics.f1_score(Y_test, Y_test_predict)}
    response = app.response_class(
        response=json.dumps({'class': str(Y_test_predict[0])}),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == '__main__':
    app.run()
