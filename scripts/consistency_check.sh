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

DOWNLOAD_DATASET_IF="OFF"

cd /workspace/$1/PaConvert/
TORCH_PROJECT_PATH=$2

echo "Installing dependencies"
python -m pip install pandas openpyxl
python -m pip install -r requirements.txt

# obtain the model test set
if [[ "$DOWNLOAD_DATASET_IF" == "ON" ]]; then
    echo '**************************start downloading datasets.....*********************************'
    mkdir -p $TORCH_PROJECT_PATH
    git clone https://github.com/open-mmlab/mmcv $TORCH_PROJECT_PATH/mmcv
    git clone https://github.com/open-mmlab/mmdetection3d $TORCH_PROJECT_PATH/mmdetection3d
    git clone https://github.com/facebookresearch/fvcore $TORCH_PROJECT_PATH/fvcore
    git clone https://github.com/fastai/timmdocs $TORCH_PROJECT_PATH/timmdocs
    git clone https://github.com/open-mmlab/mmpretrain $TORCH_PROJECT_PATH/mmpretrain
    git clone https://github.com/facebookresearch/pycls $TORCH_PROJECT_PATH/pycls
    git clone https://github.com/KevinMusgrave/pytorch-metric-learning $TORCH_PROJECT_PATH/pytorch-metric-learning
    git clone https://github.com/KaiyangZhou/deep-person-reid $TORCH_PROJECT_PATH/deep-person-reid
    git clone https://github.com/open-mmlab/mmdetection $TORCH_PROJECT_PATH/mmdetection
    git clone https://github.com/open-mmlab/mmpose $TORCH_PROJECT_PATH/mmpose
    git clone https://github.com/facebookresearch/detectron2 $TORCH_PROJECT_PATH/detectron
    git clone https://github.com/IDEA-Research/detrex $TORCH_PROJECT_PATH/detrex
    git clone https://github.com/WongKinYiu/yolov7 $TORCH_PROJECT_PATH/yolov7
    git clone https://github.com/ultralytics/yolov5 $TORCH_PROJECT_PATH/yolov5
    git clone https://github.com/open-mmlab/mmocr $TORCH_PROJECT_PATH/mmocr
    git clone https://github.com/hikopensource/DAVAR-Lab-OCR $TORCH_PROJECT_PATH/DAVAR-Lab-OCR
    git clone https://github.com/open-mmlab/mmsegmentation $TORCH_PROJECT_PATH/mmseg
    git clone https://github.com/qubvel/segmentation_models.pytorch $TORCH_PROJECT_PATH/segmentation_models
    git clone https://github.com/CSAILVision/semantic-segmentation-pytorch $TORCH_PROJECT_PATH/semantic-segmentation-pytorch
    git clone https://github.com/meetps/pytorch-semseg $TORCH_PROJECT_PATH/pytorch-semseg
    git clone https://github.com/Tramac/awesome-semantic-segmentation-pytorch $TORCH_PROJECT_PATH/awesome-semantic-segmentation-pytorch
    git clone https://github.com/PeterL1n/BackgroundMattingV2 $TORCH_PROJECT_PATH/BackgroundMattingV2
    git clone https://github.com/PeterL1n/RobustVideoMatting $TORCH_PROJECT_PATH/RobustVideoMatting
    git clone https://github.com/black0017/MedicalZooPytorch $TORCH_PROJECT_PATH/MedicalZooPytorch
    git clone https://github.com/MIC-DKFZ/nnUNet $TORCH_PROJECT_PATH/nnUNet
    git clone https://github.com/SamsungLabs/ritm_interactive_segmentation $TORCH_PROJECT_PATH/ritm_interactive_segmentation
    git clone https://github.com/allenai/allennlp $TORCH_PROJECT_PATH/allennlp
    git clone https://github.com/facebookresearch/fairseq $TORCH_PROJECT_PATH/fairseq
    git clone https://github.com/ZhangGongjie/Meta-DETR $TORCH_PROJECT_PATH/Meta-DETR
    git clone https://github.com/implus/UM-MAE $TORCH_PROJECT_PATH/UM-MAE
    git clone https://github.com/ZrrSkywalker/MonoDETR $TORCH_PROJECT_PATH/MonoDETR
    git clone https://github.com/ViTAE-Transformer/ViTPose $TORCH_PROJECT_PATH/ViTPose
    git clone https://github.com/xingyizhou/UniDet $TORCH_PROJECT_PATH/UniDet
    git clone https://github.com/facebookresearch/mae $TORCH_PROJECT_PATH/mae
    git clone https://github.com/yyliu01/PS-MT $TORCH_PROJECT_PATH/PS-MT
    git clone https://github.com/charlesCXK/TorchSemiSeg $TORCH_PROJECT_PATH/TorchSemiSeg
    git clone https://github.com/IDEA-Research/MaskDINO $TORCH_PROJECT_PATH/MaskDINO
    git clone https://github.com/HuangJunJie2017/BEVDet $TORCH_PROJECT_PATH/BEVDet
    git clone https://github.com/CASIA-IVA-Lab/Obj2Seq $TORCH_PROJECT_PATH/Obj2Seq
    git clone https://github.com/yuhangzang/OV-DETR $TORCH_PROJECT_PATH/OV-DETR
    git clone https://github.com/hikvision-research/opera $TORCH_PROJECT_PATH/opera
    git clone https://github.com/xingyizhou/CenterTrack $TORCH_PROJECT_PATH/CenterTrack
    git clone https://github.com/IDEA-Research/DINO $TORCH_PROJECT_PATH/DINO
    git clone https://github.com/pmixer/TiSASRec.pytorch $TORCH_PROJECT_PATH/TiSASRec.pytorch
    git clone https://github.com/CASIA-IVA-Lab/FastSAM $TORCH_PROJECT_PATH/FastSAM
    git clone https://github.com/graytowne/caser_pytorch $TORCH_PROJECT_PATH/caser_pytorch
    git clone https://github.com/sniklaus/pytorch-hed $TORCH_PROJECT_PATH/pytorch-hed
    git clone https://github.com/tohinz/ConSinGAN $TORCH_PROJECT_PATH/ConSinGAN
    git clone https://github.com/xinntao/Real-ESRGAN $TORCH_PROJECT_PATH/Real-ESRGAN
    git clone https://github.com/WongKinYiu/ScaledYOLOv4 $TORCH_PROJECT_PATH/ScaledYOLOv4
    git clone https://github.com/dreamquark-ai/tabnet $TORCH_PROJECT_PATH/tabnet
    git clone https://github.com/tfzhou/ContrastiveSeg $TORCH_PROJECT_PATH/ContrastiveSeg
    git clone https://github.com/facebookresearch/dino $TORCH_PROJECT_PATH/dino
    git clone https://github.com/tinyvision/DAMO-YOLO $TORCH_PROJECT_PATH/DAMO-YOLO
    git clone https://github.com/rutgerswiselab/NCR $TORCH_PROJECT_PATH/NCR
    git clone https://github.com/WenmuZhou/PAN.pytorch $TORCH_PROJECT_PATH/PAN.pytorch
    git clone https://github.com/OniroAI/MonoDepth-PyTorch $TORCH_PROJECT_PATH/MonoDepth-PyTorch
    git clone https://github.com/Lextal/pspnet-pytorch $TORCH_PROJECT_PATH/pspnet-pytorch
    git clone https://github.com/dbolya/yolact $TORCH_PROJECT_PATH/yolact
    git clone https://github.com/kazuto1011/deeplab-pytorch $TORCH_PROJECT_PATH/deeplab-pytorch
    git clone https://github.com/leeyegy/SimCC $TORCH_PROJECT_PATH/SimCC
    git clone https://github.com/ZHKKKe/MODNet $TORCH_PROJECT_PATH/MODNet
    git clone https://github.com/AkariAsai/CORA $TORCH_PROJECT_PATH/CORA
    git clone https://github.com/husthuaan/AoANet $TORCH_PROJECT_PATH/AoANet
    git clone https://github.com/SakuraRiven/EAST $TORCH_PROJECT_PATH/EAST
    git clone https://github.com/facebookresearch/detr $TORCH_PROJECT_PATH/detr
    git clone https://github.com/Tianxiaomo/pytorch-YOLOv4 $TORCH_PROJECT_PATH/pytorch-YOLOv4
    git clone https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation $TORCH_PROJECT_PATH/pytorch_Realtime_Multi-Person_Pose_Estimation
    git clone https://github.com/openai/CLIP $TORCH_PROJECT_PATH/CLIP
    git clone https://github.com/kevinzakka/recurrent-visual-attention $TORCH_PROJECT_PATH/recurrent-visual-attention
    git clone https://github.com/Fanziapril/mvfnet $TORCH_PROJECT_PATH/mvfnet
    git clone https://github.com/LINA-lln/ADDS-DepthNet $TORCH_PROJECT_PATH/ADDS-DepthNet
    git clone https://github.com/DmitryUlyanov/deep-image-prior $TORCH_PROJECT_PATH/deep-image-prior
    git clone https://github.com/facebookresearch/vilbert-multi-task $TORCH_PROJECT_PATH/vilbert-multi-task
    git clone https://github.com/ycszen/TorchSeg $TORCH_PROJECT_PATH/TorchSeg
    git clone https://github.com/gakkiri/simple-centernet-pytorch $TORCH_PROJECT_PATH/simple-centernet-pytorch
    git clone https://github.com/Shandilya21/Few-Shot $TORCH_PROJECT_PATH/Few-Shot
    git clone https://github.com/kumar-shridhar/PyTorch-BayesianCNN $TORCH_PROJECT_PATH/PyTorch-BayesianCNN
    git clone https://github.com/gpastal24/ViTPose-Pytorch $TORCH_PROJECT_PATH/ViTPose-Pytorch
    git clone https://github.com/shenweichen/DeepCTR-Torch $TORCH_PROJECT_PATH/DeepCTR-Torch
    git clone https://github.com/ZoneLikeWonderland/HACK-Model $TORCH_PROJECT_PATH/HACK-Model
    git clone https://github.com/Megvii-BaseDetection/YOLOX $TORCH_PROJECT_PATH/YOLOX
    git clone https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning $TORCH_PROJECT_PATH/a-PyTorch-Tutorial-to-Image-Captioning
    git clone https://github.com/jakesnell/prototypical-networks $TORCH_PROJECT_PATH/prototypical-networks
    git clone https://github.com/wuhuikai/FastFCN $TORCH_PROJECT_PATH/FastFCN
    git clone https://github.com/MichaelFan01/STDC-Seg $TORCH_PROJECT_PATH/STDC-Seg
    git clone https://github.com/wasidennis/AdaptSegNet $TORCH_PROJECT_PATH/AdaptSegNet
    git clone https://github.com/varunagrawal/tiny-faces-pytorch $TORCH_PROJECT_PATH/tiny-faces-pytorch
    git clone https://github.com/facebookresearch/pytorch_GAN_zoo $TORCH_PROJECT_PATH/pytorch_GAN_zoo
    git clone https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch $TORCH_PROJECT_PATH/Yet-Another-EfficientDet-Pytorch
    git clone https://github.com/NVIDIA/DeepRecommender $TORCH_PROJECT_PATH/DeepRecommender
    git clone https://github.com/RuijieJ/pren $TORCH_PROJECT_PATH/pren
    git clone https://github.com/MCG-NJU/VideoMAE $TORCH_PROJECT_PATH/VideoMAE
    git clone https://github.com/eladrich/pixel2style2pixel $TORCH_PROJECT_PATH/pixel2style2pixel
    git clone https://github.com/facebookresearch/dlrm $TORCH_PROJECT_PATH/dlrm
    git clone https://github.com/p768lwy3/torecsys $TORCH_PROJECT_PATH/torecsys
    git clone https://github.com/meliketoy/wide-resnet.pytorch $TORCH_PROJECT_PATH/wide-resnet.pytorch
    git clone https://github.com/naver-ai/pit $TORCH_PROJECT_PATH/pit
    git clone https://github.com/sacmehta/ESPNetv2 $TORCH_PROJECT_PATH/ESPNetv2
    git clone https://github.com/HobbitLong/SupContrast $TORCH_PROJECT_PATH/SupContrast
    git clone https://github.com/kenshohara/3D-ResNets-PyTorch $TORCH_PROJECT_PATH/3D-ResNets-PyTorch
    git clone https://github.com/sfatimakhan/BEIT $TORCH_PROJECT_PATH/BEIT
    git clone https://github.com/eriklindernoren/PyTorch-GAN $TORCH_PROJECT_PATH/PyTorch-GAN
    git clone https://github.com/mattmacy/vnet.pytorch $TORCH_PROJECT_PATH/vnet.pytorch
    git clone https://github.com/open-mmlab/mmaction2 $TORCH_PROJECT_PATH/mmaction2
    git clone https://github.com/ZhouHuang23/FSPNet $TORCH_PROJECT_PATH/FSPNet
    git clone https://github.com/facebookresearch/xcit $TORCH_PROJECT_PATH/xcit
    git clone https://github.com/yitu-opensource/T2T-ViT $TORCH_PROJECT_PATH/T2T-ViT
    git clone https://github.com/ycszen/ContextPrior $TORCH_PROJECT_PATH/ContextPrior
    git clone https://github.com/facebookresearch/moco $TORCH_PROJECT_PATH/moco
    git clone https://github.com/facebookresearch/moco-v3 $TORCH_PROJECT_PATH/moco-v3
    git clone https://github.com/foxlf823/sodner $TORCH_PROJECT_PATH/sodner
    git clone https://github.com/jackaduma/CycleGAN-VC2 $TORCH_PROJECT_PATH/CycleGAN-VC2
    git clone https://github.com/microsoft/unilm/tree/master/wavlm $TORCH_PROJECT_PATH/wavlm
    git clone https://github.com/microsoft/unilm/tree/master/markuplm $TORCH_PROJECT_PATH/markuplm
    git clone https://github.com/plkmo/BERT-Relation-Extraction $TORCH_PROJECT_PATH/BERT-Relation-Extraction
    git clone https://github.com/liusongxiang/ppg-vc $TORCH_PROJECT_PATH/ppg-vc
    git clone https://github.com/daniilrobnikov/vits2 $TORCH_PROJECT_PATH/vits2
    git clone https://github.com/jjery2243542/adaptive_voice_conversion $TORCH_PROJECT_PATH/adaptive_voice_conversion
    git clone https://github.com/liusongxiang/StarGAN-Voice-Conversion $TORCH_PROJECT_PATH/StarGAN-Voice-Conversion
    git clone https://github.com/jaywalnut310/glow-tts $TORCH_PROJECT_PATH/glow-tts
    git clone https://github.com/yhcc/utcie $TORCH_PROJECT_PATH/utcie
    git clone https://github.com/microsoft/speecht5 $TORCH_PROJECT_PATH/speecht5
    git clone https://github.com/maum-ai/voicefilter $TORCH_PROJECT_PATH/voicefilter
    git clone https://github.com/J-zin/SNUH $TORCH_PROJECT_PATH/SNUH
    git clone https://github.com/cao-hu/oneee $TORCH_PROJECT_PATH/oneee
    git clone https://github.com/WendellGul/DCMH $TORCH_PROJECT_PATH/DCMH
    git clone https://github.com/amirveyseh/definition_extraction $TORCH_PROJECT_PATH/definition_extraction
    git clone https://github.com/ljynlp/w2ner $TORCH_PROJECT_PATH/w2ner
    git clone https://github.com/facebookresearch/xformers $TORCH_PROJECT_PATH/xformers
    git clone https://github.com/princeton-nlp/PURE $TORCH_PROJECT_PATH/PURE
    git clone https://github.com/yujiapingyu/Deep-Hashing $TORCH_PROJECT_PATH/Deep-Hashing
    git clone https://github.com/yuanli2333/Hadamard-Matrix-for-hashing $TORCH_PROJECT_PATH/Hadamard-Matrix-for-hashing
    git clone https://github.com/allenai/primer $TORCH_PROJECT_PATH/primer
fi

echo '************************************************************************************'
echo '**************************start converting code set*********************************'
failed_projects=()
for project in "$TORCH_PROJECT_PATH"/*; do
    if [ -d "$project" ]; then
        project_name=$(basename "$project")
        echo "Converting project: $project_name"
        python paconvert/main.py --in_dir "$project" --show_unsupport_api --calculate_speed
        if [ $? -ne 0 ]; then
            failed_projects+=("$project_name")
        fi
    fi
done

check_error1=$(( ${#failed_projects[@]} > 0 ? 1 : 0 ))

echo '************************************************************************************'
echo '**************************start converting common API case**************************'
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
    echo "Your PR code-test-set (more than 20W+ lines) convert check failed."
    echo "The following projects failed to convert:"
    printf '%s\n' "${failed_projects[@]}"
else
    echo "Your PR code-test-set (more than 20W+ lines) convert check passed."
fi

if [ ${check_error2} != 0  ]; then  
    echo "Your PR code example convert check failed."
else
    echo "Your PR code example convert check passed."
fi
echo -e '************************************************************************************'

check_error=$((check_error1||check_error2))
exit ${check_error}
