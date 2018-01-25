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
import csv

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

csv_dic = dict()

# with open('/Users/pk4634/Documents/phase2/harvey_temporal_data_tsv/30_aug_31920_tsv.csv') as fline0:
with open('/Users/pk4634/Documents/phase2/irma_temporal_data_tsv/crees_labelled/noduplicate/overall_noduplicate/08_sep_1480_label.csv') as fline0:
    for line0 in csv.reader(fline0, delimiter='\t', skipinitialspace='True', quotechar=None):
        if csv_dic.has_key(line0[0]):
            print ''
        else:
            csv_dic[line0[0]] = line0
# with open('/Users/pk4634/Documents/phase2/harvey_temporal_data_tsv/crisc_labelled/semantics/05_sep_1619_label.csv') as fline,\
#        open('/Users/pk4634/Documents/phase2/harvey_temporal_data_tsv/crisc_labelled/30_aug_31920_label.csv', 'wb') as wfile:

with open('/Users/pk4634/Documents/phase2/irma_temporal_data_tsv/crisc_labelled/noduplicate/semantics/08_sep_1480_sem_X.csv') as fline,\
        open('/Users/pk4634/Documents/phase2/irma_temporal_data_tsv/crisc_labelled/noduplicate/08_sep_1480_label.csv', 'wb') as wfile:

    writer = csv.writer(wfile, delimiter='\t')
    c = 0
    for line in csv.reader(fline, delimiter='\t', skipinitialspace='True', quotechar=None):
        c = c + 1
        if c > 0:
            print c
            '''enr_tweet = line[1]  # ["enr_tweet"]
            Token_count = line[15]  # ["Token_count"]
            TweetL = line[13]  # ["TweetL"]
            Verbs = line[12]  # ["Verbs"]
            Nouns = line[11]  # ["Nouns"]
            HashTag = line[16]  # ["HashTag"]
            Pronouns = line[14]  # ["Pronouns"]
            Readability = line[10]  # ["Readability"]'''

            enr_tweet = line[2]  # ["enr_tweet"]
            Token_count = line[19]  # ["Token_count"]
            TweetL = line[17]  # ["TweetL"]
            Verbs = line[16]  # ["Verbs"]
            Nouns = line[15]  # ["Nouns"]
            HashTag = line[20]  # ["HashTag"]
            Pronouns = line[18]  # ["Pronouns"]
            Readability = line[14]  # ["Readability"]

            data_test = np.array([[enr_tweet, Nouns, Verbs, Pronouns,
                                   TweetL, Token_count, HashTag, Readability]])

            text_str = line[1]
            if enr_tweet.strip() == '':

                label_str = 'not-applicable'
                writer.writerow([line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7],
                                 line[8], line[9], line[10], line[11], line[12], line[13], label_str])

            else:
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

                label_str = ''

                if Y_test_predict == 1.0:
                    label_str = 'related'
                if Y_test_predict == 0.0:
                    label_str = 'non-related'

            # print Y_test_predict[0]
                if csv_dic.has_key(line[0]):

                    tweet_list = csv_dic[line[0]]
                    writer.writerow([line[0], line[1], tweet_list[2], line[3], line[4], line[5], line[6], line[7],
                                     line[8], line[9], line[10], line[11], line[12], line[13], label_str])

    wfile.close()
