import cv2
import time
from face_compare import Compare

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

c.del_cuda_ctx()