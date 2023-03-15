import argparse
import os
import re
import time

import requests
import json
import base64
import numpy as np
import cv2
from gevent import pywsgi
from threading import Thread

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
CORS(app, supports_credentials=True)


def base64_to_numpy(image_base64):
    image_bytes = base64.b64decode(image_base64)
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image_np2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image_np2


@app.route("/callback", methods=["POST"])
def recv_data():
    data = request.get_data(as_text=True)
    print(data)
    return ""


def demo(video_url, callback_port, requests_url):
    input_json = {
        "video_url": video_url,
        "blacklist_path": "http://192.168.1.135:9002/api/scalper/callBack/black",
        "callback": "http://127.0.0.1:{}/callback".format(callback_port),
        "update_blacklist": False
    }
    response = requests.post(requests_url, headers=headers, json=input_json)
    print(response)
    print("Status code:", response.status_code)
    res = json.loads(response.text)
    print(res)


def run(callback_port):
    server = pywsgi.WSGIServer(('0.0.0.0', int(callback_port)), app)
    print('Listening on address: 0.0.0.0:{}'.format(callback_port))
    server.serve_forever()


parser = argparse.ArgumentParser()
parser.add_argument('--port', type=str, default=6006)
parser.add_argument('--callback_port', type=int, default=35002)
arg = parser.parse_args()

requests_url = "http://127.0.0.1:{}/recognize".format(arg.port)
headers = {
    'Connection': 'close',
}

t = Thread(target=run, args=(arg.callback_port,))
t.start()
demo("http://192.168.1.203:8080/live/0B065FD5DA3ED4E4C3EE54EF95FCF030.live.mp4?vhost=__defaultVhost__", arg.callback_port, requests_url)
print("send")
