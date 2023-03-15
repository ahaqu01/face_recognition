import json
import time
import pycuda.driver as cuda
import numpy as np
import os
import cv2
import h5py
import base64
from sklearn.metrics.pairwise import pairwise_distances

import torch

from config.config import env_cfg, inference_cfg, model_cfg, thresholds
from retinaface.src.face_detect import face_detector
from facenet.src.cal_embedding import cal_face_embedding
import requests
import logging


class Compare:
    def __init__(self,
                 blacklist_path="",
                 blacklist_update=True,
                 rebuild=True,
                 blacklist_callback=""
                 ):
        """
        blacklist_path: 黑名单路径
        """
        # load cfgs
        self.gpu_id = env_cfg["gpu_id"]
        self.face_det_model_path = inference_cfg["face_det_model_path"]
        self.face_cog_model_path = inference_cfg["face_cog_model_path"]
        self.features_lib_dir = inference_cfg["features_lib_dir"]
        self.max_face_num = inference_cfg["max_face_num"]
        self.det_speedup = inference_cfg["det_speed_up"]
        self.cog_speedup = inference_cfg["cog_speed_up"]
        self.speed_up_weights = inference_cfg["speed_up_weights"]
        self.face_cog_model_name = model_cfg["face_cog_model_name"]
        self.metric = model_cfg["metric"]  # choose from ["cosine", "euclidean", "euclidean_l2"]

        self.rebuild_engine = rebuild

        # set model device
        self.device = torch.device('cuda:{}'.format(self.gpu_id) if torch.cuda.is_available() else 'cpu')
        cuda.init()
        self.cuda_ctx = cuda.Device(int(self.gpu_id)).make_context()

        # 加载retinaface模型
        self.face_det = face_detector(
            backbone_name="mobile0.25",
            model_weights=self.face_det_model_path,
            keep_size=False,
            confidence_threshold=0.3,
            top_k=5000,
            nms_threshold=0.4,
            keep_top_k=100,
            vis_thres=0.7,
            device=self.device,
            speed_up=self.det_speedup,
            speed_up_weights=self.speed_up_weights,
            rebuild_engine=self.rebuild_engine,
        )
        print("model of face detection finished!")

        # 加载facenet512模型
        self.face_cog = cal_face_embedding(
            model_weights=self.face_cog_model_path,
            device=self.device,
            speed_up_weights=self.speed_up_weights,
            speed_up=self.cog_speedup,
            max_face_num=self.max_face_num,
            rebuild_engine=self.rebuild_engine,
        )
        print("model of face recognition finished!")

        # face cog thr
        self.threshold = thresholds[self.face_cog_model_name][self.metric]

        # 初始化黑名单特征
        # self.blacklist_face_det = inference_cfg["blacklist_face_det"]
        self.blacklist_ids, self.blacklist_face_embeddings, self.blacklist_features_lib_path = \
            self.init_lib_features(blacklist_path, self.features_lib_dir, blacklist_update, blacklist_callback)

    def del_cuda_ctx(self):
        self.cuda_ctx.pop()
        del self.cuda_ctx

    # 读取黑名单接口
    def _read_blacklist(self, blacklist_url, black_callback):
        blacklist_ids = []
        blacklist_face_embeddings = []
        # the blacklist_ids' item type is str
        # the blacklist_face_embeddings' item type is ndarray

        res_obj = {}  # 回调数据
        # 请求黑名单接口
        try:
            blacklist_data = requests.get(blacklist_url)
        except Exception as e:
            # 请求失败
            res_obj['err_code'] = 2
            res_obj['err_message'] = "request blacklist fail"
            # 发送callback
            res = requests.post(black_callback, json=res_obj)
        else:
            blacklist_json = json.loads(blacklist_data.text)
            if blacklist_json['code'] == 0:
                # 获取黑名单数据
                for item in blacklist_json['data']:
                    person_id = item['globalPersonId']
                    for image in item['faceImage']:
                        try:
                            # base64解码
                            img_bytes = base64.b64decode(image)
                            image_np = np.frombuffer(img_bytes, dtype=np.uint8)

                            assert list(image_np) != []
                            # 转换为np.ndarray
                            person_img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
                            h, w = person_img.shape[:2]
                            person_img = cv2.copyMakeBorder(person_img, h//2, h//2, w//2, w//2, cv2.BORDER_CONSTANT, value=(128,128,128))

                            assert person_img is not None
                            # 人脸检测
                            dets = self.face_det.inference_single(person_img)

                            assert list(dets) != []
                            # 获得最大人脸bbox
                            face_areas = [(det[3] - det[1]) * (det[2] - det[0]) for det in dets]
                            face_max_idx = np.argmax(np.array(face_areas))
                            det_max = dets[face_max_idx]
                            # 提取最大人脸embedding
                            face_max_embedding = self.face_cog.get_img_faces_embedding(person_img, [det_max])
                            blacklist_ids.append(person_id)
                            blacklist_face_embeddings.append(face_max_embedding)

                        except Exception as e:
                            pass

                if len(blacklist_ids) == len(blacklist_face_embeddings) == 0:
                    # 请求成功，黑名单无数据
                    res_obj['err_code'] = 0
                    res_obj['err_message'] = "blacklist is empty"
                    # 发送callback
                    res = requests.post(black_callback, json=res_obj)
            else:
                # 黑名单接口返回码异常
                res_obj['err_code'] = 1
                res_obj['err_message'] = "blacklist error with code {}".format(blacklist_json['code'])
                # 发送callback
                res = requests.post(black_callback, json=res_obj)

        # blacklist_paths = [os.path.join(blacklist_url, img_name) for img_name in os.listdir(blacklist_url)]
        # for blacklist_path in blacklist_paths:
        #     person_id = blacklist_path.split("/")[-1].split(".")[0]
        #     person_img = cv2.imread(blacklist_path)
        #     # 人脸检测
        #     dets = self.face_det.inference_single(person_img)
        #     # 获得最大人脸bbox
        #     face_areas = [(det[3] - det[1]) * (det[2] - det[0]) for det in dets]
        #     face_max_idx = np.argmax(np.array(face_areas))
        #     det_max = dets[face_max_idx]
        #     # 提取最大人脸embedding
        #     face_max_embedding = self.face_cog.get_img_faces_embedding(person_img, [det_max])
        #     blacklist_ids.append(person_id)
        #     blacklist_face_embeddings.append(face_max_embedding)

        if len(blacklist_face_embeddings) == 0:
            blacklist_face_embeddings = np.array(blacklist_face_embeddings)
        else:
            blacklist_face_embeddings = np.concatenate(blacklist_face_embeddings, 0)
        logging.info("read {} faces from blacklist lib".format(len(blacklist_ids)))
        return blacklist_ids, blacklist_face_embeddings

    # 建立黑名单特征库
    def _save_features_lib(self, blacklist_ids, blacklist_face_embeddings, blacklist_features_lib_path):
        # blacklist_face_embeddings is ndarray
        with h5py.File(blacklist_features_lib_path, "w") as f_w:
            for i, k in enumerate(blacklist_ids):
                if str(k) not in f_w.keys():
                    f_w.create_group(str(k)).create_dataset("0", data=blacklist_face_embeddings[i])
                else:
                    f_w[str(k)].create_dataset(str(len(f_w[str(k)])), data=blacklist_face_embeddings[i])

        logging.info("save features lib succeed")

    # 读取黑名单特征库
    def _read_features_lib(self, blacklist_features_lib_path):
        blacklist_ids = []
        blacklist_face_embeddings = []

        with h5py.File(blacklist_features_lib_path, "r") as f_r:
            for key in f_r.keys():
                for item in f_r[key]:
                    blacklist_ids.append(key)
                    blacklist_face_embeddings.append(f_r[key][item][()])

        return blacklist_ids, blacklist_face_embeddings

    # 初始化特征库
    def init_lib_features(self, blacklist_path, features_lib_dir, blacklist_update, black_callback):
        os.makedirs(features_lib_dir, exist_ok=True)
        blacklist_features_lib_path = os.path.join(features_lib_dir, "features.h5")

        # 如果不存在features.h5或请求更新，则重新建立特征库
        if not os.path.exists(blacklist_features_lib_path) or blacklist_update:
            logging.info("start build {}".format(blacklist_features_lib_path))
            blacklist_ids, blacklist_face_embeddings = self._read_blacklist(blacklist_path, black_callback)
            # 保存到文件
            if len(blacklist_ids) != 0:
                self._save_features_lib(blacklist_ids, blacklist_face_embeddings, blacklist_features_lib_path)
        else:
            logging.info("start read {}".format(blacklist_features_lib_path))
            blacklist_ids, blacklist_face_embeddings = self._read_features_lib(blacklist_features_lib_path)
            if len(blacklist_face_embeddings) != 0:
                blacklist_face_embeddings = np.stack(blacklist_face_embeddings, 0)

        # blacklist_ids, blacklist_face_embeddings = self._read_blacklist(blacklist_path)
        # print(len(blacklist_ids), len(blacklist_face_embeddings))
        # print(blacklist_ids)
        return blacklist_ids, blacklist_face_embeddings, blacklist_features_lib_path

    # 更新特征库
    def update_lib_features(self, blacklist_path, features_lib_dir, black_callback):
        os.makedirs(features_lib_dir, exist_ok=True)
        blacklist_features_lib_path = os.path.join(features_lib_dir, "features.h5")

        logging.info("start update {}".format(blacklist_features_lib_path))
        blacklist_ids, blacklist_face_embeddings = self._read_blacklist(blacklist_path, black_callback)
        # 加载到内存
        self.blacklist_ids = blacklist_ids.copy()
        self.blacklist_face_embeddings = blacklist_face_embeddings.copy()
        # 保存到文件
        if len(blacklist_ids) != 0:
            self._save_features_lib(blacklist_ids, blacklist_face_embeddings, blacklist_features_lib_path)

    def cal_dists(self, face_embeddings):
        def l2_normalize(x):
            x_dim = x.shape[1]
            x_normed = x / np.tile(np.expand_dims(np.linalg.norm(x, axis=1), -1), (1, x_dim))
            return x_normed

        if self.metric == "cosine" or self.metric == "euclidean":
            dists = pairwise_distances(face_embeddings, Y=self.blacklist_face_embeddings, metric=self.metric)
        elif self.metric == "euclidean_l2":
            dists = pairwise_distances(X=l2_normalize(face_embeddings),
                                       Y=l2_normalize(self.blacklist_face_embeddings),
                                       metric="euclidean")
        else:
            print("metric is not supported!")
            dists = None
        return dists

    def dists_post_process(self, dists):
        response_list = []
        face_num = dists.shape[0]
        for face_idx in range(face_num):
            dist = dists[face_idx]
            person_ID = self.blacklist_ids[np.argmin(dist)]
            person_minimum = dist[np.argmin(dist)]
            if person_minimum < self.threshold:
                person_info = {
                    "detect_idx": face_idx,
                    "face_ID": person_ID,
                    "distance": person_minimum,
                }
                response_list.append(person_info)
        return response_list

    def single_frame_inference(self, img):
        # img, ndarray, BGR
        img = img.copy()

        # 人脸检测
        dets = self.face_det.inference_single(img)
        # 删除不合法bbox
        boxes, pred = self.face_det.del_irregular_bboxes(dets, img)
        print("detected {} faces".format(len(boxes)))

        if len(boxes) == 0:
            return []
        else:
            # 黑名单不为空才检测
            #if len(self.blacklist_face_embeddings) != 0:
            # 人脸特征提取
            face_embeddings = self.face_cog.get_img_faces_embedding(img, boxes)
            dists = self.cal_dists(face_embeddings)

            if dists is not None:
                res = self.dists_post_process(dists)
                return res
            else:
                return []

            # else:
            #     logging.info("pass recognition: blacklist is empty")
            #     return []


if __name__ == "__main__":
    c = Compare(
        blacklist_path="/workspace/face_recognition/blacklist",
        blacklist_update=True
    )
    img_path = "/workspace/face_recognition/test_img/tangyan2.jpg"
    img = cv2.imread(img_path)

    repeat_time = 50
    t_s = time.time()
    for i in range(repeat_time):
        res = c.single_frame_inference(img)
    t_e = time.time()
    print((t_e - t_s) / repeat_time)
    print(res)
