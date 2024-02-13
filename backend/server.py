from flask import Flask
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['POST'])
def main():
    json_data = json.dumps(0)
    response = app.response_class(
        response=json_data,
        status=200,
        mimetype='application/json'
    )

    return response


if __name__ == '__main__':
    app.run(host = '127.0.0.1', port = 5001, debug = True)