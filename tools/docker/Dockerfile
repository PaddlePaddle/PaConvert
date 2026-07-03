# A image for testing PaConvert
FROM registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.8-cudnn8.6-trt8.5-gcc82

ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN apt-get -o Acquire::Retries=5 update || true; \
    apt-get -o Acquire::Retries=5 install -y net-tools || true

RUN ln -sf `which python3.10` /usr/bin/python

RUN ln -sf `which pip3.10` /usr/local/bin/pip

RUN python -m pip install --no-cache-dir \
        astor \
        pandas \
        openpyxl \
        pytest-timeout \
        pytest-rerunfailures \
        pytest-xdist

RUN wget -q https://paddle-paconvert.bj.bcebos.com/torch-2.7.1+cu118-cp310-cp310-manylinux_2_28_x86_64.whl \
            https://paddle-paconvert.bj.bcebos.com/torchvision-0.22.1+cu118-cp310-cp310-manylinux_2_28_x86_64.whl && \ 
    python -m pip install --no-cache-dir \
        torch-2.7.1+cu118-cp310-cp310-manylinux_2_28_x86_64.whl \
        torchvision-0.22.1+cu118-cp310-cp310-manylinux_2_28_x86_64.whl && \
    rm -f torch-2.7.1+cu118-cp310-cp310-manylinux_2_28_x86_64.whl \
          torch-2.7.1+cu118-cp310-cp310-manylinux_2_28_x86_64.whl

RUN wget --tries=5 --timeout=60 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb

RUN apt-get -o Acquire::Retries=5 update || true; \
    apt-get -o Acquire::Retries=5 install -y cudnn9-cuda-11
