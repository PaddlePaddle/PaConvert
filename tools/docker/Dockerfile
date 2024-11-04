# A image for testing PaConvert
FROM registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.8-cudnn8.6-trt8.5-gcc82

RUN apt-get update && \
    apt-get install -y net-tools

RUN ln -sf `which python3.10` /usr/bin/python

RUN ln -sf `which pip3.10` /usr/local/bin/pip

RUN python -m pip install astor 

RUN python -m pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 检查Ubuntu版本
RUN cat /etc/os-release | grep VERSION_ID
# 获取Ubuntu版本和架构信息
RUN export DISTRO=$(. /etc/os-release; echo $ID$VERSION_ID | sed 's/[^0-9]//g') && \
    export ARCH=$(dpkg --print-architecture) && \
    echo "Detected distro: $DISTRO" && \
    echo "Detected architecture: $ARCH"
