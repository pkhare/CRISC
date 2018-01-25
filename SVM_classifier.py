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


'''file_str = 'tweets_stat_filtered_SPQL_new_level3n7_annotationlemma_hypernym_semantics.arff'

data = arff.load(open(
    '/Users/pk4634/Documents/new data/recurssive_exp/relatedNOTinformative/ColoradoFlood/all_train_' + file_str, 'rb'))'''

file_str = 'tweets_stat_filtered_SPQL_new_level3n7_annotationlemma_hypernym_semantics.arff'

data = arff.load(open(
    '/Users/pk4634/Documents/new data/recurssive_exp/relatedNOTinformative/all_events_' + file_str, 'rb'))

x = data['data']

data1 = np.array(x)
X = data1[:, 0:8]
Y = data1[:, 8].astype(np.float32)

t = ['Document', 'NumberOfNouns', 'NumberOfVerbs', 'NumberOfPronouns',
     'TweetLength', 'NumberOfWords', 'NumberOfHashTag', 'Readability']

frm = pd.DataFrame(X, columns=t)


vectorizer = CountVectorizer(analyzer='word', tokenizer=tokenize_and_stem,
                             stop_words='english', lowercase=True, ngram_range=(1, 1), max_features=20000)
doc_vectorize = vectorizer.fit_transform(frm.Document)
tf_transform = TfidfTransformer()
tf_vectorize = tf_transform.fit_transform(doc_vectorize)

X_data = sp.sparse.hstack((tf_vectorize, frm[['NumberOfNouns', 'NumberOfVerbs', 'NumberOfPronouns', 'TweetLength',
                                              'NumberOfWords', 'NumberOfHashTag', 'Readability']].values.astype(np.float32)), format='csr')
X_data.toarray().shape

#############################
svc = SVC(kernel='linear', degree=3, gamma='auto', tol=0.001)

svc.fit(X_data, Y)

filename = "SVC_classifier_model_9events.pkl"

pickle.dump(svc, open(filename, 'wb'))

pickle.dump(vectorizer, open("vectorizer_9events.pkl", "wb"), pickle.HIGHEST_PROTOCOL)
pickle.dump(tf_transform, open("tf_transform_9events.pkl", "wb"), pickle.HIGHEST_PROTOCOL)

#############################
'''
data_test = arff.load(open('/Users/pk4634/Documents/new data/recurssive_exp/relatedNOTinformative/ColoradoFlood/test_'+file_str,'rb'))
x_test = data_test['data']


data_test = np.array(x_test)
X_test = data_test[:,0:8]
Y_test = data_test[:,8].astype(np.float32)

t = ['Document','NumberOfNouns','NumberOfVerbs','NumberOfPronouns','TweetLength','NumberOfWords','NumberOfHashTag','Readability']

frm_test = pd.DataFrame(X_test, columns=t)
#frm_test

test_vectorize = vectorizer.transform(frm_test.Document)
test_tf_vectorize = tf_transform.transform(test_vectorize)

X_data_test = sp.sparse.hstack((test_tf_vectorize,frm_test[['NumberOfNouns','NumberOfVerbs','NumberOfPronouns','TweetLength','NumberOfWords','NumberOfHashTag','Readability']].values.astype(np.float32) ),format='csr')

#X_data_test.toarray().shape


#Y_test_predict = svc.predict(X_data_test)

#print metrics.f1_score(Y_test,Y_test_predict)
loaded_model = pickle.load(open(filename,"rb"))
#result = loaded_model.score(X_data_test, Y_test)
Y_test_predict = loaded_model.predict(X_data_test)
print (metrics.f1_score(Y_test,Y_test_predict))'''
