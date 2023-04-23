from ensemble import run_ensemble
from flask import Flask, request
from flask_cors import CORS, cross_origin

# initialize flask server
app = Flask(__name__)
CORS(app)

# default route (unused)
@app.route('/')
def index():
    return 'Index Page'

# upload route for processing and returning interpretation
@app.route('/upload', methods=['GET'])
def upload():
    # process video
    res = run_ensemble()
    # return interpretation
    return res
    
# run flask server
if __name__ == "__main__":
    app.run(port=8000, debug=True)
