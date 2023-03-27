import asyncio
import os
import sys
from subprocess import call

import snoop
from ensemble import run_ensemble
from flask import Flask, request
from flask_cors import CORS, cross_origin

# sys.path.append('/flask-server/ensemble')
# sys.path.append(os.path.abspath('ensemble'))
# from ensemble.ensemble import run_ensemble
# import ensemble

app = Flask(__name__)
CORS(app)
# app.config['CORS_HEADERS'] = 'no-cors'

@app.route('/')
def index():
    return 'Index Page'


@app.route('/upload', methods=['GET'])
def upload():
    # hard code path to video uploaded
    # process video
    # run_ensemble()
    res = run_ensemble()
   
    # once we have received interpreted results, delete video so only one video is in folder at a time
    # return interpretation
    return res
    

if __name__ == "__main__":
    app.run(port=8000, debug=True)
