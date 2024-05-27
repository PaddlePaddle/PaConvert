# A image for testing PaConvert
FROM registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.8-cudnn8.6-trt8.5-gcc82

RUN apt-get update && \
    apt-get install -y net-tools

RUN ln -sf `which python3.9` /usr/bin/python

RUN ln -sf `which pip3.9` /usr/local/bin/pip

RUN python -m pip install astor 

RUN python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
