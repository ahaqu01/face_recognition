FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

WORKDIR /workspace/face_recognition/

COPY *.py /workspace/face_recognition/
COPY facenet/ /workspace/face_recognition/facenet/
COPY retinaface/ /workspace/face_recognition/retinaface/
COPY config/ /workspace/face_recognition/config/
# COPY speed_up_weights/ /workspace/face_recognition/speed_up_weights/
COPY requirements.txt /workspace/face_recognition/
COPY TensorRT-8.0.0.3/ /home/TensorRT-8.0.0.3/
COPY start.sh /workspace/face_recognition/

ENV PATH /usr/local/cuda-10.2/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda-10.2/lib64:$LD_LIBRARY_PAT
ENV LD_LIBRARY_PATH /home/TensorRT-8.0.0.3/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH /home/TensorRT-8.0.0.3/lib:$LIBRARY_PATH

EXPOSE 6006

RUN apt-key adv --recv-keys --keyserver keyserver.ubuntu.com A4B469963BF863CC \
	&& apt-get update \
	&& apt-get install -y libgl1-mesa-glx \
	&& apt-get install -y libglib2.0-dev

RUN cd /home/TensorRT-8.0.0.3 \
	&& pip install python/tensorrt-8.0.0.3-cp37-none-linux_x86_64.whl \
	&& pip install uff/uff-0.6.9-py2.py3-none-any.whl \
	&& cd /workspace/face_recognition \
	&& pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# COPY blacklist/ /workspace/face_recognition/blacklist/
