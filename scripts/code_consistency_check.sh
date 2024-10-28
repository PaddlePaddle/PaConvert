# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set +x

export FLAGS_set_to_1d=0
DOWNLOAD_DATASET_IF="OFF"

cd /workspace/$1/PaConvert/
TORCH_PROJECT_PATH=$2

echo "Insalling latest release cpu version torch"
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print('torch version information:' ,torch.__version__)"

echo "Insalling develop cpu version paddle"
python -m pip uninstall -y paddlepaddle
python -m pip uninstall -y paddlepaddle-gpu
rm -rf /root/anaconda3/lib/python*/site-packages/paddlepaddle-0.0.0.dist-info/
python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
python -c "import paddle; print('paddle version information:' , paddle.__version__); commit = paddle.__git_commit__;print('paddle commit information:' , commit)"

echo "Installing dependencies"
python -m pip install pandas openpyxl

# obtain the model test set
if [[ "$DOWNLOAD_DATASET_IF" == "ON" ]]; then
    echo '**************************start downloading datasets.....*********************************'
    mkdir -p $TORCH_PROJECT_PATH
    git clone https://github.com/open-mmlab/mmcv.git $TORCH_PROJECT_PATH/mmcv
    git clone https://github.com/open-mmlab/mmdetection3d.git $TORCH_PROJECT_PATH/mmdetection3d
    git clone https://github.com/facebookresearch/fvcore.git $TORCH_PROJECT_PATH/fvcore
    git clone https://github.com/fastai/timmdocs.git $TORCH_PROJECT_PATH/timmdocs
    git clone https://github.com/open-mmlab/mmpretrain.git $TORCH_PROJECT_PATH/mmpretrain
    git clone https://github.com/facebookresearch/pycls.git $TORCH_PROJECT_PATH/pycls
    git clone https://github.com/KevinMusgrave/pytorch-metric-learning.git $TORCH_PROJECT_PATH/pytorch-metric-learning
    git clone https://github.com/KaiyangZhou/deep-person-reid.git $TORCH_PROJECT_PATH/deep-person-reid
    git clone https://github.com/open-mmlab/mmdetection.git $TORCH_PROJECT_PATH/mmdetection
    git clone https://github.com/open-mmlab/mmpose.git $TORCH_PROJECT_PATH/mmpose
    git clone https://github.com/facebookresearch/detectron2.git $TORCH_PROJECT_PATH/detectron
    git clone https://github.com/IDEA-Research/detrex.git $TORCH_PROJECT_PATH/detrex
    git clone https://github.com/WongKinYiu/yolov7.git $TORCH_PROJECT_PATH/yolov7
    git clone https://github.com/ultralytics/yolov5.git $TORCH_PROJECT_PATH/yolov5
    git clone https://github.com/open-mmlab/mmocr.git $TORCH_PROJECT_PATH/mmocr
    git clone https://github.com/hikopensource/DAVAR-Lab-OCR.git $TORCH_PROJECT_PATH/DAVAR-Lab-OCR
    git clone https://github.com/open-mmlab/mmsegmentation.git $TORCH_PROJECT_PATH/mmseg
    git clone https://github.com/qubvel/segmentation_models.pytorch.git $TORCH_PROJECT_PATH/segmentation_models
    git clone https://github.com/CSAILVision/semantic-segmentation-pytorch.git $TORCH_PROJECT_PATH/semantic-segmentation-pytorch
    git clone https://github.com/meetps/pytorch-semseg.git $TORCH_PROJECT_PATH/pytorch-semseg
    git clone https://github.com/Tramac/awesome-semantic-segmentation-pytorch.git $TORCH_PROJECT_PATH/awesome-semantic-segmentation-pytorch
    git clone https://github.com/PeterL1n/BackgroundMattingV2.git $TORCH_PROJECT_PATH/BackgroundMattingV2
    git clone https://github.com/PeterL1n/RobustVideoMatting.git $TORCH_PROJECT_PATH/RobustVideoMatting
    git clone https://github.com/black0017/MedicalZooPytorch.git $TORCH_PROJECT_PATH/MedicalZooPytorch
    git clone https://github.com/MIC-DKFZ/nnUNet.git $TORCH_PROJECT_PATH/nnUNet
    git clone https://github.com/SamsungLabs/ritm_interactive_segmentation.git $TORCH_PROJECT_PATH/ritm_interactive_segmentation
    git clone https://github.com/allenai/allennlp.git $TORCH_PROJECT_PATH/allennlp
    git clone https://github.com/facebookresearch/fairseq.git $TORCH_PROJECT_PATH/fairseq
    git clone https://github.com/ZhangGongjie/Meta-DETR.git $TORCH_PROJECT_PATH/Meta-DETR
    git clone https://github.com/implus/UM-MAE.git $TORCH_PROJECT_PATH/UM-MAE
    git clone https://github.com/ZrrSkywalker/MonoDETR.git $TORCH_PROJECT_PATH/MonoDETR
    git clone https://github.com/ViTAE-Transformer/ViTPose.git $TORCH_PROJECT_PATH/ViTPose
    git clone https://github.com/xingyizhou/UniDet.git $TORCH_PROJECT_PATH/UniDet
    git clone https://github.com/facebookresearch/mae.git $TORCH_PROJECT_PATH/mae
    git clone https://github.com/yyliu01/PS-MT.git $TORCH_PROJECT_PATH/PS-MT
    git clone https://github.com/charlesCXK/TorchSemiSeg.git $TORCH_PROJECT_PATH/TorchSemiSeg
    git clone https://github.com/IDEA-Research/MaskDINO.git $TORCH_PROJECT_PATH/MaskDINO
    git clone https://github.com/HuangJunJie2017/BEVDet.git $TORCH_PROJECT_PATH/BEVDet
    git clone https://github.com/CASIA-IVA-Lab/Obj2Seq.git $TORCH_PROJECT_PATH/Obj2Seq
    git clone https://github.com/yuhangzang/OV-DETR.git $TORCH_PROJECT_PATH/OV-DETR
    git clone https://github.com/hikvision-research/opera.git $TORCH_PROJECT_PATH/opera
    git clone https://github.com/xingyizhou/CenterTrack.git $TORCH_PROJECT_PATH/CenterTrack
    git clone https://github.com/IDEA-Research/DINO.git $TORCH_PROJECT_PATH/DINO
fi

# Check the grammar mechanism of the test set and other issues
echo '**************************start converting test case********************************'
python paconvert/main.py --in_dir $TORCH_PROJECT_PATH --show_unsupport 1;check_error1=$?
echo '************************************************************************************'
#check whether common API transfer is successful

echo '**************************start converting common API case********************************'
python tools/consistency/consistency_check.py;check_error2=$?


echo '************************************************************************************'
echo "______      _____                          _   "
echo "| ___ \    / ____|                        | |  "
echo "| |_/ /_ _| |     ___  _ ____   _____ _ __| |_ "
echo "|  __/ _  | |    / _ \\| \\_ \\ \\ / / _ \\ \\__| __|"
echo "| | | (_| | |___| (_) | | | \\ V /  __/ |  | |_ "
echo "\\_|  \\__,_|\\_____\\___/|_| |_|\\_/ \\___|_|   \\__|"
echo -e '\n************************************************************************************'

if [ ${check_error1} != 0  ]; then  
    echo "Your PR code-test-set (more than 15W+ lines) convert check failed."
else
    echo "Your PR code-test-set (more than 15W+ lines) convert check passed."
fi

if [ ${check_error2} != 0  ]; then  
    echo "Your PR code example convert check failed."
else
    echo "Your PR code example convert check passed."
fi
echo -e '************************************************************************************'

check_error=$((check_error1||check_error2))
exit ${check_error}
