# A image for testing PaConvert
FROM registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.8-cudnn8.6-trt8.5-gcc82

RUN apt-get update && \
    apt-get install -y net-tools

RUN python3 -m pip install astor

RUN python3 -m pip install install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
