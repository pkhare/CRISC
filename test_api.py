import requests
import json
import numpy as np
import pandas as pd

tweet = 'Fire activity has intensified around the area of #Bell. #Dargan Follow @NSWRFS @nswpolice @FireRescueNSW #BlueMountains #NSWFires'
payload = {'tweet': tweet}

api_response = requests.get('http://localhost:4567/crisc', params=payload)
if(api_response.ok):
    json_reponse = json.loads(api_response.content)
    '''for key in json_reponse:
        print key + " : " + str(json_reponse[key])'''

    '''print json_reponse["enr_tweet"]
    print json_reponse["Token_count"]
    print json_reponse["TweetL"]
    print json_reponse["Verbs"]
    print json_reponse["Nouns"]
    print json_reponse["HashTag"]
    print json_reponse["Pronouns"]
    print json_reponse["Readability"]'''

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
    t = ['Document', 'NumberOfNouns', 'NumberOfVerbs', 'NumberOfPronouns',
         'TweetLength', 'NumberOfWords', 'NumberOfHashTag', 'Readability']

    print data_test
    # print data_test.shape
    # print data_test[:, 0:8]
    X_test = data_test[:, 0:8]

    frm_test = pd.DataFrame(X_test, columns=t)
