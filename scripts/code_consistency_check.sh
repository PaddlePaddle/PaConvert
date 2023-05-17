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

DOWNLOAD_DATASET_IF="OFF"
DEVELOP_IF="OFF"
TORCH_PROJECT_PATH="torch_project"

if [[ "$DEVELOP_IF" == "OFF" ]]; then
    cd /workspace/$3/PaConvert/
    PATH=$1
    TORCH_PROJECT_PATH=$2
fi

# obtain the test case
if [[ "$DOWNLOAD_DATASET_IF" == "ON" ]]; then
    mkdir -p torch_project
    git clone https://github.com/open-mmlab/mmcv.git torch_project/mmcv
    git clone https://github.com/open-mmlab/mmdetection3d.git torch_project/mmdetection3d
    git clone https://github.com/facebookresearch/fvcore.git torch_project/fvcore
    git clone https://github.com/fastai/timmdocs.git torch_project/timmdocs
    git clone https://github.com/open-mmlab/mmpretrain.git torch_project/mmpretrain
    git clone https://github.com/facebookresearch/pycls.git torch_project/pycls
    git clone https://github.com/KevinMusgrave/pytorch-metric-learning.git torch_project/pytorch-metric-learning
    git clone https://github.com/KaiyangZhou/deep-person-reid.git torch_project/deep-person-reid
    git clone https://github.com/open-mmlab/mmdetection.git torch_project/mmdetection
    git clone https://github.com/open-mmlab/mmpose.git torch_project/mmpose
    git clone https://github.com/facebookresearch/detectron2.git torch_project/detectron
    git clone https://github.com/IDEA-Research/detrex.git torch_project/detrex
    git clone https://github.com/WongKinYiu/yolov7.git torch_project/yolov7
    git clone https://github.com/ultralytics/yolov5.git torch_project/yolov5
    git clone https://github.com/open-mmlab/mmocr.git torch_project/mmocr
    git clone https://github.com/hikopensource/DAVAR-Lab-OCR.git torch_project/DAVAR-Lab-OCR
    git clone https://github.com/open-mmlab/mmsegmentation.git torch_project/mmseg
    git clone https://github.com/qubvel/segmentation_models.pytorch.git torch_project/segmentation_models
    git clone https://github.com/CSAILVision/semantic-segmentation-pytorch.git torch_project/semantic-segmentation-pytorch
    git clone https://github.com/meetps/pytorch-semseg.git torch_project/pytorch-semseg
    git clone https://github.com/Tramac/awesome-semantic-segmentation-pytorch.git torch_project/awesome-semantic-segmentation-pytorch
    git clone https://github.com/PeterL1n/BackgroundMattingV2.git torch_project/BackgroundMattingV2
    git clone https://github.com/PeterL1n/RobustVideoMatting.git torch_project/RobustVideoMatting
    git clone https://github.com/black0017/MedicalZooPytorch.git torch_project/MedicalZooPytorch
    git clone https://github.com/MIC-DKFZ/nnUNet.git torch_project/nnUNet
    git clone https://github.com/SamsungLabs/ritm_interactive_segmentation.git torch_project/ritm_interactive_segmentation
    git clone https://github.com/allenai/allennlp.git torch_project/allennlp
    git clone https://github.com/facebookresearch/fairseq.git torch_project/fairseq
    git clone https://github.com/ZhangGongjie/Meta-DETR.git torch_project/Meta-DETR
    git clone https://github.com/implus/UM-MAE.git torch_project/UM-MAE
    git clone https://github.com/ZrrSkywalker/MonoDETR.git torch_project/MonoDETR
    git clone https://github.com/ViTAE-Transformer/ViTPose.git torch_project/ViTPose
    git clone https://github.com/xingyizhou/UniDet.git torch_project/UniDet
    git clone https://github.com/facebookresearch/mae.git torch_project/mae
    git clone https://github.com/yyliu01/PS-MT.git torch_project/PS-MT
    git clone https://github.com/charlesCXK/TorchSemiSeg.git torch_project/TorchSemiSeg
    git clone https://github.com/IDEA-Research/MaskDINO.git torch_project/MaskDINO
    git clone https://github.com/HuangJunJie2017/BEVDet.git torch_project/BEVDet
    git clone https://github.com/CASIA-IVA-Lab/Obj2Seq.git torch_project/Obj2Seq
    git clone https://github.com/yuhangzang/OV-DETR.git torch_project/OV-DETR
    git clone https://github.com/hikvision-research/opera.git torch_project/opera
    git clone https://github.com/xingyizhou/CenterTrack.git torch_project/CenterTrack
    git clone https://github.com/IDEA-Research/DINO.git torch_project/DINO
fi

# Check the grammar mechanism of the test set and other issues
echo '**************************start converting test case********************************'
python paconvert/main.py --in_dir $TORCH_PROJECT_PATH;check_error1=$?
echo '************************************************************************************'
#check whether common API transfer is successful

echo '**************************start converting common API case********************************'
mkdir tests/code_library/code_case/temp_paddle_code
python tools/consistency/api_code_consistency_check.py;check_error2=$?
rm -rf tests/code_library/code_case/temp_paddle_code

echo '************************************************************************************'
echo "______                                   _   "
echo "| ___ \                                 | |  "
echo "| |_/ /_ _  ___ ___  _ ____   _____ _ __| |_ "
echo "|  __/ _  |/ __/ _ \\| \_ \ \ / / _ \ \__| __|"
echo "| | | (_| | (_| (_) | | | \\ V /  __/ |  | |_ "  
echo "\\_|  \\__,_|\\___\\___/|_| |_|\\_/ \\___|_|   \\__|"  
echo -e '\n************************************************************************************'
if [ ${check_error1} != 0  ]; then  
    echo "Your PR code test set translation check failed."
else
    echo "Your PR code test set translation check passed."
fi

if [ ${check_error2} != 0  ]; then  
    echo "Your PR code common code translation check failed."
else
    echo "Your PR code common code translation check passed."
fi
echo -e '************************************************************************************'

exit ${check_error1}&&${check_error2}
