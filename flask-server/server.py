from ensemble import run_ensemble
from flask import Flask, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return 'Index Page'


@app.route('/upload', methods=['GET'])
def upload():
    # process video
    res = run_ensemble()

    # return interpretation
    return res
    

if __name__ == "__main__":
    app.run(port=8000, debug=True)
