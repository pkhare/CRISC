import xgboost
from flask import Flask
from flask_restful import Resource, Api
from json import dumps
import arff
import sys
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
api = Api(app)

################################

stemmer = PorterStemmer()


def tokenize_and_stem(text):
    #tokens = word_tokenize(text)
    tokens = regexp_tokenize(text, pattern=r"\s|[\.,:;'()?!]", gaps=True)
    # strip out punctuation and make lowercase
    tokens = [token.lower().strip(string.punctuation)
              for token in tokens if token.isalnum()]

    # now stem the tokens
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens

################################


filename = "SVC_classifier_model.pkl"
loaded_model = pickle.load(open(filename, "rb"))

filename_vectorize = "vectorizer.pkl"
filename_tf_transform = "tf_transform.pkl"

vectorizer = pickle.load(open(filename_vectorize, "rb"))
tf_transform = pickle.load(open(filename_tf_transform, "rb"))
################################


class Crisc_model(Resource):

    def get(self, file_str):

        ################################

        #file_str = 'tweets_stat_filtered_SPQL_new_level3n7_annotationlemma_hypernym_semantics.arff'
        data_test = arff.load(open(
            '/Users/pk4634/Documents/new data/recurssive_exp/relatedNOTinformative/ColoradoFlood/test_' + file_str, 'rb'))

        x_test = data_test['data']

        data_test = np.array(x_test)
        X_test = data_test[:, 0:8]
        Y_test = data_test[:, 8].astype(np.float32)
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
        print (metrics.f1_score(Y_test, Y_test_predict))

        # return {'departments': 'yess'}
        return {'departments': metrics.f1_score(Y_test, Y_test_predict)}


api.add_resource(Crisc_model, '/crisc/<string:file_str>')

if __name__ == '__main__':
    app.run()
