# 典型案例展示--Llama大模型的转换

## 步骤1: 依赖安装与源代码库下载

### 安装依赖项

在开发本项目之前，请确保已经安装了以下依赖项：
- paconvert
- Python 3.8+
- paddlepaddle-gpu 2.6+ (建议: develop)
- paddlenlp

### 源代码库下载

Llama源码下载命令
```bash
 git clone https://github.com/meta-llama/llama.git
```

## 步骤2: 模型代码转换

模型代码转换使用如下命令：

```python
paconvert --in_dir /Llama/path --output_dir output/path --log_dir my_log/path
```
Llama模型已实现一键转换，故无需手动编写转换规则，只需指定输入路径和输出路径即可。但对于其他待转模型可能存在未转换情形，欢迎参考[贡献手册](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md)向本项目贡献代码。

## 步骤3: 模型参数转换：

Pytorch 的模型参数与 Paddle 的模型参数无法共用，本案例中有两种方式获取适用于 Paddle 的模型参数。

### 使用 Pytorch 权重进行转化

PaddleNLP 提供了可自动将 PyTorch 相关的权重转化为 Paddle 权重的接口，代码如下：

```python
from paddlenlp.transformers import AutoModelForCausalLM

AutoModelForCausalLM.from_pretrained("/path/to/pytorch/model", convert_from_torch=True, dtype="float16")
```

> dtype 为转化权重的真实 dtype 数据类型，通常为：float16, bloat16 和 float32。

以上代码可自动加载 pytorch 权重并转化为对应 paddle 权重保存在 `/path/to/pytorch/model` 目录下。

更多细节可参考 [模型格式转换](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/community/contribute_models/convert_pytorch_to_paddle.rst)

### 直接下载权重

从[AI Studio](https://aistudio.baidu.com/modelsdetail/636/space)下载Llama模型参数。


## 步骤4：运行转换后代码

使用如下命令运行转换后的代码：
```python
CUDA_VISIBLE_DEVICES=0 python -m paddle.distributed.launch --devices=0 /example_chat_completion/path/example_chat_completion.py --ckpt_dir /model_param/path --tokenizer_path /tokenizer/path/tokenizer.model --max_seq_len 64

# 运行结果
# Mayonnaise is a sauce made from egg yolk, vinegar, lemon juice, and oil.
# ...

```
[可选] example_chat_completion.py 中输入的对话列表 `dialogs` 有多条对话，若机器显存有限，可删除部分对话，以节省显存。
