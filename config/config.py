env_cfg = {
    "gpu_id": "0",
}

inference_cfg = {
    "face_det_model_path": "/workspace/face_recognition/retinaface/src/weights/mobilenet0.25_Final.pth",
    "face_cog_model_path": "/workspace/face_recognition/facenet/src/weights/20180402-114759-vggface2.pt",
    "features_lib_dir": "./features_lib",
    "max_face_num": 20,
    "det_speed_up": True,
    "cog_speed_up": True,
    "rebuild_engine": False,
    "speed_up_weights": "/workspace/face_recognition/speed_up_weights",
    "blacklist_face_det": True,
}

blacklist_cfg = {
    "blacklist_path": "http://192.168.1.135:9002/api/scalper/callBack/black",
    "blacklist_callback": ""
}

model_cfg = {
    "face_cog_model_name": "Facenet512",
    "metric": "euclidean_l2",
}

thresholds = {
    'VGG-Face': {'cosine': 0.40, 'euclidean': 0.60, 'euclidean_l2': 0.86},
    'Facenet': {'cosine': 0.40, 'euclidean': 10, 'euclidean_l2': 0.80},
    'Facenet512': {'cosine': 0.30, 'euclidean': 23.56, 'euclidean_l2': 1.04},
    'ArcFace': {'cosine': 0.68, 'euclidean': 4.15, 'euclidean_l2': 1.13},
    'Dlib': {'cosine': 0.07, 'euclidean': 0.6, 'euclidean_l2': 0.4},

    # TODO: find the best threshold values
    'SFace': {'cosine': 0.5932763306134152, 'euclidean': 10.734038121282206, 'euclidean_l2': 1.055836701022614},

    'OpenFace': {'cosine': 0.10, 'euclidean': 0.55, 'euclidean_l2': 0.55},
    'DeepFace': {'cosine': 0.23, 'euclidean': 64, 'euclidean_l2': 0.64},
    'DeepID': {'cosine': 0.015, 'euclidean': 45, 'euclidean_l2': 0.17}

}
