# -*- coding: utf-8 -*-
import os
import sys
import requests
from json import dumps
import json

tweet = "texas fellas, is it gay to be in this hurricane? i mean, you're literally getting blown by a dude named harvey..... sound a lil spicy 2 me"
# RT @DavidLammy: If you can help with clothes, food, blankets, toiletries etc please donate to: St Clements Church, 95 Sirdar Rd, W11 4EQ #G…
# please do not waste food: donate to the needy and contact NGO
# the smoke affected the entire residential area, you cannot really breath
# I got a bunch of clothes I’d like to donate to hurricane sandy victims. Anyone know where/how I can do that?
payload = {'twstr': tweet}

api_response = requests.get('http://localhost:5000/crisc', params=payload)
if(api_response.ok):
    json_reponse = json.loads(api_response.content)
    print json_reponse
