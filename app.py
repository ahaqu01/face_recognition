import argparse
import json
import os
import cv2
import logging
import time
from logging.handlers import TimedRotatingFileHandler

from functools import wraps

import requests
from flask import Flask, request, make_response, jsonify
from flask_cors import CORS
from gevent import pywsgi
from threading import Thread
from face_compare import Compare
from config.config import inference_cfg, blacklist_cfg
import ctypes
import inspect

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
CORS(app, supports_credentials=True)

global_blacklist_callback = blacklist_cfg['blacklist_callback']
global_blacklist_path = blacklist_cfg['blacklist_path']
# global_blacklist_path = blacklist_cfg['blacklist_path']
thread_pool = []


# 使用多线程进行异步调用
def asyncc(f):
    global thread_pool
    wraps(f)

    def wrapper(*args, **kwargs):
        thr = Thread(target=f, args=args, kwargs=kwargs)
        thread_pool.append(thr)
        thr.start()

    return wrapper


# 结束线程
def async_raise(tid, exctype):
    tid = ctypes.c_long(tid)
    print(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def log(logfolder):
    if not os.path.exists(logfolder):
        os.makedirs(logfolder)
    # 初始化logging
    logging.basicConfig()
    logger = logging.getLogger()
    # 设置日志级别
    logger.setLevel(logging.INFO)
    # 添加TimeRoatingFileHandler
    # 定义一个1天换一次log文件的handler
    # 保留7个旧log文件
    timefilehandler = TimedRotatingFileHandler(os.path.join(logfolder, "log.log"), when='D',
                                               interval=1, backupCount=7, encoding="utf-8", )
    timefilehandler.suffix = "%Y-%m-%d.log"
    # 设置log记录输出的格式
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(filename)s-%(lineno)d-%(message)s')
    timefilehandler.setFormatter(formatter)
    # 添加到logger中
    logger.addHandler(timefilehandler)


# @asyncc
def response_callback(result, callback, status_code):
    response_obj = {
        'status_code': status_code,
        'scalper_num': len(result)
    }

    # 检测正常
    if status_code == 0:
        if len(result) >= 1:
            face_ids = []
            for res in result:
                face_ids.append(res['face_ID'])
                res['distance'] = str(res['distance'])

            response_obj['scalper_id'] = face_ids
            response_obj['recognize_results'] = result
        try:
            res = requests.post(callback, json=response_obj)
            logging.info("callback success {}".format(response_obj))
        except Exception as e:
            logging.info("callback fail with error {}".format(e))

    # 检测异常
    elif status_code == 1:
        try:
            res = requests.post(callback, json=response_obj)
            logging.info("callback success {}".format(response_obj))
        except Exception as e:
            logging.info("callback fail with error {}".format(e))


@asyncc
def face_recognize_main(jsonobj):
    global global_blacklist_path

    try:
        url = jsonobj['video_url']
        blacklist = jsonobj['blacklist_path']
        is_update = jsonobj['update_blacklist']
        cog_callback = jsonobj['recognition_callback']
        black_callback = jsonobj['blacklist_callback']
    except:
        logging.info("Error in input parameters")

    else:
        # is_update = True if update_blacklist == "True" else False

        # 如果黑名单接口发生变化或主动请求更新黑名单，则重新建立特征库
        if blacklist != global_blacklist_path or is_update:

            global_blacklist_path = blacklist
            try:
                comp.update_lib_features(
                    blacklist_path=blacklist,
                    features_lib_dir=inference_cfg['features_lib_dir'],
                    black_callback=black_callback
                )
                logging.info("succeed update blacklist {}".format(blacklist))
            except Exception as e:
                logging.warning("fail update blacklist with error {}".format(e))

        if len(comp.blacklist_ids) != 0:
            try:
                cap = cv2.VideoCapture(url)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                res, frame = cap.read()
                logging.info("succeed read video {}, fps:{}".format(url, fps))
            except Exception as e:
                logging.warning("fail read video {} with error {}".format(url, e))
            else:
                count = 1
                interval = int(fps / 2)  # 每隔几帧检测一次
                while res and count <= 2 * 60 * fps:
                    if count % interval == 0:
                        try:
                            result = comp.single_frame_inference(frame)
                            status_code = 0
                        except Exception as e:
                            result = []
                            status_code = 1
                            logging.warning("fail process a frame with error {}".format(e))
                        finally:
                            if len(result) > 0:
                                response_callback(result, cog_callback, status_code)
                    count += 1
                    res, frame = cap.read()
                logging.info("finish one segment")
        else:
            logging.info("pass one segment because blacklist is empty")


@app.route("/recognize", methods=["POST"])
def run():
    global thread_pool

    result = {
        "status_code": 0,
    }
    try:
        data = request.get_data(as_text=True)
        logging.info("get data: {}".format(data))
        result["status_code"] = 200
        jsonobj = json.loads(data)
        # assert ['video_url', 'blacklist_path', 'callback', 'update_blacklist'] == list(jsonobj.keys()), "Error in input parameters"

    except Exception as e:
        logging.error("get data error: {}".format(str(e)))
        result["status_code"] = 500
        result["err_message"] = "get data error: {}".format(str(e))

    else:
        # 旧线程未结束则杀死线程
        if len(thread_pool) > 0:
            for t in thread_pool:
                if t.is_alive():
                    try:
                        async_raise(t.ident, SystemExit)
                        logging.info("receive new request, kill old thread")
                    except Exception as e:
                        logging.warning("kill old thread fail with error {}".format(e))
            thread_pool.clear()
        # 异步调用
        face_recognize_main(jsonobj)

    response = make_response(jsonify(result), 200)

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='logs/', help='logs filefolder')
    parser.add_argument('--port', type=int, default='6006', help='listening port')
    parser.add_argument('--rebuild', action="store_true", help='whether rebuild model')
    arg = parser.parse_args()

    log(arg.log)
    try:
        comp = Compare(
            blacklist_path=global_blacklist_path,
            blacklist_update=False,
            rebuild=arg.rebuild,
            blacklist_callback=global_blacklist_callback
        )

        logging.info("******* 服务启动 ********")
        server = pywsgi.WSGIServer(('0.0.0.0', int(arg.port)), app)
        logging.info('Listening on address: 0.0.0.0:{}'.format(arg.port))
        server.serve_forever()

    except Exception as e:
        logging.warning("start fail with error {}".format(e))
