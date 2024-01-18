# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install git -y
RUN ln -sf /usr/share/zoneinfo/Australia/Brisbane /etc/localtime
RUN apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl
RUN apt-get install -y llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv

ENV LANG C.UTF-8
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH $PYENV_ROOT/bin:/root/.pyenv/versions/3.9.18/bin:/usr/bin:/usr/local/nvidia/bin:/usr/local/nvidia/lib64:$PATH
RUN eval "$(pyenv init --path)"

RUN git --version
RUN pyenv --version

RUN pyenv install 3.9.18
RUN pyenv global 3.9.18

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PYTHON_VERSION=3.9.18

RUN pip install --upgrade pip
RUN pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
#COPY requirements.txt .
RUN which nvcc
#
#COPY ./ ./
#RUN pip install -r requirements.txt
#RUN pip install -e .
#RUN ./make_assuming_venv_activated.sh
#RUN python3 ./deformable_detr/models/ops/test.py

# RUN pip install gsutil
#RUN nvcc --version
#RUN git --version
#RUN python --version
#RUN pyenv versions
#RUN ls -la /root/.pyenv/versions/3.9.18/bin
#RUN which python
#RUN which python3
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python3", "./deformable_detr/models/ops/test.py"]
# CMD ["python", "./deformable_detr/models/ops/test.py"]
# # Create working directory
# RUN mkdir -p /usr/src/app
# WORKDIR /usr/src/app
#
# # Copy contents
# COPY . /usr/src/app






# Copy weights
#RUN python3 -c "from models import *; \
#attempt_download('weights/yolov5s.pt'); \
#attempt_download('weights/yolov5m.pt'); \
#attempt_download('weights/yolov5l.pt')"


# ---------------------------------------------------  Extras Below  ---------------------------------------------------

# Build and Push
# t=ultralytics/yolov5:latest && sudo docker build -t $t . && sudo docker push $t
# for v in {300..303}; do t=ultralytics/coco:v$v && sudo docker build -t $t . && sudo docker push $t; done

# Pull and Run
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host $t

# Pull and Run with local directory access
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all -v "$(pwd)"/coco:/usr/src/coco $t

# Kill all
# sudo docker kill $(sudo docker ps -q)

# Kill all image-based
# sudo docker kill $(sudo docker ps -a -q --filter ancestor=ultralytics/yolov5:latest)

# Bash into running container
# sudo docker container exec -it ba65811811ab bash

# Bash into stopped container
# sudo docker commit 092b16b25c5b usr/resume && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco --entrypoint=sh usr/resume

# Send weights to GCP
# python -c "from utils.general import *; strip_optimizer('runs/exp0_*/weights/best.pt', 'tmp.pt')" && gsutil cp tmp.pt gs://*.pt

# Clean up
# docker system prune -a --volumes
