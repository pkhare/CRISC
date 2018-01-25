#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 21:03:55 2017

@author: pk4634
"""

import scipy
import numpy
import sklearn
import xgboost
from flask import Flask
from flask_restful import Resource, Api
from json import dumps

app = Flask(__name__)
api = Api(app)


@app.route('/')
def hello_world():
    return 'Hello World!\n'


class Departments_Meta(Resource):
    def get(self, department_name):
        return {'departments': 'yess'}


api.add_resource(Departments_Meta, '/departments/<string:department_name>')

if __name__ == '__main__':
    app.run()
