# 典型案例展示--Llama大模型的转换

## 步骤1: 依赖安装与源代码库下载

### 安装依赖项

在开发本项目之前，请确保已经安装了以下依赖项：
- paconvert
- Python 3.8+
- paddlepaddle-gpu 2.6+ (建议: develop)
- paddleformers
- wget

### 源代码库下载

Llama源码下载命令
```bash
 git clone https://github.com/meta-llama/llama.git
```

## 步骤2: 模型代码转换

模型代码转换使用如下命令：

```python
paconvert -i ./Llama -o ./convert_model/Llama
```
Llama模型已实现一键转换，故无需手动编写转换规则，只需指定输入路径和输出路径即可。但对于其他待转模型可能存在未转换情形，欢迎参考[贡献手册](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md)向本项目贡献代码。

## 步骤3: 模型参数获取：

Pytorch 的模型参数与 Paddle 的模型参数无法共用，可用如下命令获取适用于 Paddle 的模型参数。

```bash
cd ./convert_model/Llama
wget https://x2paddle.bj.bcebos.com/PaConvert/llama-7B/LLama-7B-Weights.tar.gz
tar -xvf LLama-7B-Weights.tar.gz
# LLama-7B-Weights/params.json
# LLama-7B-Weights/paddle_llama.00.pth
# LLama-7B-Weights/tokenizer.model
```

如需手动转换原始 Pytorch 权重可参考[模型格式转换](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/zh/community/contribute_models/convert_pytorch_to_paddle.rst)。


## 步骤4：运行转换后代码

使用如下命令运行转换后的代码：
```python
CUDA_VISIBLE_DEVICES=0 python -m paddle.distributed.launch --devices=0 ./example_chat_completion.py --ckpt_dir ./LLama-7B-Weights --tokenizer_path ./LLama-7B-Weights/tokenizer.model --max_seq_len 64

# 运行结果
# Mayonnaise is a sauce made from egg yolk, vinegar, lemon juice, and oil.
# ...

```
[可选] example_chat_completion.py 中输入的对话列表 `dialogs` 有多条对话，若机器显存有限，可删除部分对话，以节省显存。
