# A image for testing PaConvert
FROM registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.8-cudnn8.6-trt8.5-gcc82

RUN apt-get update && \
    apt-get install -y net-tools

RUN ln -sf `which python3.10` /usr/bin/python

RUN ln -sf `which pip3.10` /usr/local/bin/pip

RUN python -m pip install astor 

RUN python -m pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb

RUN apt-get update && \
    apt-get install -y \
    cudnn9-cuda-11

