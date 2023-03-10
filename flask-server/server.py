from flask import Flask, request
from flask_cors import CORS, cross_origin

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

    # once we have received interpreted results, delete video so only one video is in folder at a time
    # return interpretation
    return {"video": ["video1", "video2", "video3"]}
    

if __name__ == "__main__":
    app.run(port=8000, debug=True)