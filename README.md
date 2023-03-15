### face recognition

#### 一、环境配置
见requirements.txt
由于加速需要，需额外安装tensorrt
安装方法参考官网：
https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-debian

（内部人员可使用face_recognition:latest镜像的环境）

构建镜像所需TensorRT包官网下载地址：https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.0/tars/TensorRT-8.0.0.3.Linux.x86_64-gnu.cuda-10.2.cudnn8.2.tar.gz

#### 二、使用样例

git clone --recurse-submodules  -b yyc_dev  https://github.com/ahaqu01/face_recognition.git

下载代码

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

#### 三、参数配置
首次启动使用--rebuild参数初始化加速模型，例如：

```
python app.py --rebuild
```

或

```
bash start.sh --rebuild
```

#### 四、加速细节
face detector：优化了retinaface的数据处理方式，使用TensorRT加速引擎

face recognizer：使用TensorRT加速引擎

特征比对过程：for循环->矩阵运算

其他代码上的一些细节优化

目前在test.py中，一次inference时间为90ms以内（原先300ms以上）

