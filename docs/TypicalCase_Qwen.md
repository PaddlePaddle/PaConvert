# 典型案例展示--Qwen大模型的转换

## 步骤1: 依赖安装与源代码库下载

### 安装依赖项

在开发本项目之前，请确保已经安装了以下依赖项：
- paconvert
- Python 3.8+
- paddlepaddle-gpu 2.6+ (建议: develop)
- paddleformers

### 源代码库下载

Qwen源码下载命令
```bash
 git clone https://huggingface.co/Qwen/Qwen-7B-Chat
```

## 步骤2: 模型代码转换

模型代码转换使用如下命令：

```python
paconvert -i ./Qwen-7B-Chat -o ./convert_model/Qwen-7B-Chat
```
Qwen模型已实现一键转换，故无需手动编写转换规则，只需指定输入路径和输出路径即可。但对于其他待转模型可能存在未转换情形，欢迎参考[贡献手册](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md)向本项目贡献代码。

## 步骤3: 模型参数获取：

Pytorch 的模型参数与 Paddle 的模型参数无法共用，[AI Studio](https://aistudio.baidu.com/modelsdetail/666/space) 提供了Qwen模型参数的下载入口，相关命令如下：

```bash
git lfs install
git clone http://git.aistudio.baidu.com/aistudio/qwen-7b-chat.git
# 使用 model.safetensors.index.json, model-0000x-of-00004.safetensors 替换 ./convert_model/Qwen-7B-Chat 目录下的 model.safetensors.index.json, model-0000x-of-00008.safetensors。
```

如需手动转换原始 Pytorch 权重可参考[模型格式转换](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/en/community/contribute_models/convert_pytorch_to_paddle.rst)。

## 步骤4：手动转换部分配置文件

当前，部分LLM(Large Language Model)的config配置文件可能需要手动修改以适配PaddlePaddle。

### 1. 修改config文件

在`config.json`中新增配置项`"dtype"`用于指明当前模型参数类型，帮助内存分配器合理的分配合适空间。本例中需增加如下配置：

```json
"dtype": "bfloat16"
```

### [可选] 2.修改转换后的代码

`torch.nn.functional.scaled_dot_product_attention` 对应 `paddle.nn.functional.scaled_dot_product_attention`，但paddle的后端实现要求GPU计算能力不低于8.0，但torch并无此要求，当GPU计算能力低于8.0时，需手动转换部分代码。本例中可将`SUPPORT_TORCH2`设置为`False`，避免使用`torch.nn.functional.scaled_dot_product_attention`分支。

```python
SUPPORT_TORCH2 = False
```

## 步骤5：运行转换后代码

### 1. 新建python文件
在 ./convert_model/Qwen-7B-Chat 目录下，创建一个名为 `run_qwen.py` 的文件，内容如下：

```python
import paddle
from modeling_qwen import QWenLMHeadModel
from tokenization_qwen import QWenTokenizer

tokenizer = QWenTokenizer.from_pretrained("./")

model = QWenLMHeadModel.from_pretrained("./")

# 第一轮对话 1st dialogue turn
response, history = model.chat(tokenizer, "你好", history=None)
# 你好！有什么我可以帮助你的吗？

# 第二轮对话 2nd dialogue turn
response, history = model.chat(tokenizer, "给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
# 当然可以，这是一个关于一位叫李晓明的年轻人的故事。他出生在一个普通的家庭，但他从小就有着梦想，希望能成为一名成功的商人。\n\n李# 晓明在高中毕业后就开始了自己的创业之路。他最初选择做的是开一家小超市，但是由于经营不善，他的商店很快就破产了。不过，李晓明并没# 有因此而放弃，而是从失败中吸取教训，并且重新振作起来，开始了新的创业之旅。\n\n这次，李晓明选择了做电商，因为他发现这是一个具有# 巨大潜力的行业。他努力学习和研究电商知识，不断提高自己的技能。他还通过网络社交平台寻找客户，并且提供优质的商品和服务，得到了客# 户的认可和支持。\n\n经过几年的努力，李晓明的电商公司终于取得了成功，他的销售额每年都在稳步增长。他也成为了一名备受尊敬的企业# # 家，并且被社会上的人们所熟知和尊重。\n\n这个故事告诉我们，只要我们有梦想、有毅力、肯付出努力，就一定能够实现自己的目标。无论面# 临多大的困难，我们都不能轻易放弃，要坚持到底，相信自己一定能够成功。

# 第三轮对话 3rd dialogue turn
response, history = model.chat(tokenizer, "给这个故事起一个标题", history=history)
# 《从失败到成功：李晓明的创业经历》
```
### 2. 运行代码

```python
python -m paddle.distributed.launch ./convert_model/Qwen-7B-Chat/run_qwen.py

```
