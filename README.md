# 概述
paddleconverter是一个代码转换工具，能自动将使用其它深度学习框架训练或推理的代码转换为使用PaddlePaddle的代码，方便代码迁移，更好地使用PaddlePaddle的特性。

目前支持转换Pytorch代码，其它深度学习框架逐步增加中，原理是通过Python AST语法树分析，将输入代码生成为抽象语法树，对其进行解析、遍历、匹配、编辑、替换、插入等各种操作，然后得到基于PaddlePaddle的抽象语法树，最后生成Paddle的代码。

转换会尽量保持原代码的风格与结构，将代码中调用其它深度学习框架的接口转换为调用PaddlePaddle的接口。

转换过程中不改动原文件，会将原项目文件一一转换到 `out_dir` 指定的文件夹中，方便前后对比调试。同时对不同类型的文件会分别处理：

- Python代码文件：识别代码中调用其它深度学习框架的接口并转换
- requirements.txt： 替换其中的安装依赖为 `paddlepaddle-gpu`
- 其他文件：原样拷贝

# 安装与使用

由于使用了一些较新的Python语法树特性，你需要使用>=python3.8的解释器。

1. 使用pip安装

```bash
python3.8 -m pip install -U paconvert
paconvert -in_dir torch_project [--out_dir paddle_project] [--exclude_dirs exclude_dirs] [--log_dir log_dir] [--log_level "DEBUG"] [--run_check 1]
```

2. 使用源码安装

```bash
git clone https://github.com/PaddlePaddle/PaConvert.git
python3.8 paconvert/main.py --in_dir torch_project [--out_dir paddle_project] [--exclude_dirs exclude_dirs] [--log_dir log_dir] [--log_level "DEBUG"] [--run_check 1]
```

**参数介绍**

```
参数：
--in_dir        输入torch项目文件，可以为单个文件或文件夹
--out_dir       可选，输出paddle项目文件，可以为单个文件或文件夹，默认在当前目录下创建./paddle_project/
--exclude_dirs  可选，排除转换的文件或文件夹，多个项目请用逗号分隔，默认无
--log_dir       可选，输出日志的路径，默认会在当前目录下创建convert.log
--log_level     可选"INFO" "DEBUG"，打印log等级，默认"INFO"
--run_check     可选，工具自检
```


# 示例

下面以输入pytorch代码进行转换为例：
```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.Linear as Linear
import torch.nn.functional as F

class MyNet(nn.Module):
    test = "str"

    def __init__(self):
        self._fc1 = torch.nn.Linear(10, 10)
        self._fc2 = nn.Linear(10, 10)
        self._fc3 = Linear(10, 10)

    @torch.no_grad()
    def forward(self, x):
        x = self._fc1(x)
        x = self._fc2(x)
        x = self._fc3(x)
        y = torch.add(x, x)
        return F.relu(y)

net = MyNet()

sgd = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
lr = optim.lr_scheduler.MultiStepLR(sgd, milestones=[2, 4, 6], gamma=0.8)


```

转换完成后：
```
import paddle


class MyNet(paddle.nn.Layer):
    test = 'str'

    def __init__(self):
        self._fc1 = paddle.nn.Linear(in_features=10, out_features=10)
        self._fc2 = paddle.nn.Linear(in_features=10, out_features=10)
        self._fc3 = paddle.nn.Linear(in_features=10, out_features=10)

    @paddle.no_grad()
    def forward(self, x):
        x = self._fc1(x)
        x = self._fc2(x)
        x = self._fc3(x)
        y = paddle.add(x=x, y=x)
        return paddle.nn.functional.relu(x=y)


net = MyNet()
>>>sgd = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
>>>lr = torch.optim.lr_scheduler.MultiStepLR(sgd, milestones=[2, 4, 6], gamma=0.8)

```

打印信息如下：

```txt
===========================================
PyTorch to Paddle Convert Start ------>:
===========================================
Start convert /workspace/example_code.py --> /workspace/PaConvert/paddle_code/example_code.py
[example_code.py:1] remove 'import torch' 
[example_code.py:2] remove 'import torch.nn as nn' 
[example_code.py:3] remove 'import torch.optim as optim' 
[example_code.py:4] remove 'import torch.nn.Linear as Linear' 
[example_code.py:5] remove 'import torch.nn.functional as F' 
[example_code.py] add 'import paddle' in first line
[example_code.py:25] [Not Support] can not convert torch.optim.SGD to Paddle 
[example_code.py:26] [Not Support] can not convert torch.optim.lr_scheduler.MultiStepLR to Paddle 
Finish convert /workspace/example_code.py --> /workspace/PaConvert/paddle_code/example_code.py


========================================
Convert Summary:
========================================
There are 10 Pytorch APIs in this Project:
 8  Pytorch APIs have been converted to Paddle successfully!
 2  Pytorch APIs are not supported to convert currently!
 Convert Rate is: 80.00%

For these 2 Pytorch APIs that do not support Convert, which have been marked by >>> before the line. Please refer to https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html and convert it by yourself manually.

Thank you to use Paddle Convert tool. You can make any suggestions to us.

```

转换完成后，会打印 **转换总结** ，包含 **API总数、转换成功数、未转换数、转换率** 。如未指定 `out_dir` ，则会在当前目录下 `./paddle_project/` 并输出到改目录。如未指定 `log_dir` ，则会在当前目录下创建 `convert.log` 并保存与终端相同内容的日志。

例如，上述代码里一共有10个Pytorch API，其中8个被成功转换，因此转换率为 `80.00%` ，如果项目中有多个python文件，则会统计所有文件的累计数据。

对于转换成功的API，代码风格上会略有变化，会 **补全API全名、补全参数关键字、移除注释、移除多余空行** 。因为在 `源码->语法树->源码` 的过程中，会采用标准写法来生成代码，而注释、空行等代码无法被语法树识别，将被移除。

对于未转换的API，将 **补全为Pytorch API全名**，同时在行前通过 `>>>` 的形式加以标记，用户必须对该API进行手动转换，可参考[Pytorch-Paddle API映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html#pytorch-1-13-paddle-2-4-api)，然后删除标记，否则代码无法运行。

# 额外说明

注意在转换pytorch代码时，当前仅转换Pytorch原生的API，对于基于Pytorch API封装的其他第三方库（例如`mmdet`、`mmcv`、`torchvision`、`torchaudio`等），目前无法转换，这部分API依赖人工转换。我们后续会考虑逐步支持。

转换时工具会尽可能保证代码行数不变，但在某些情形下可能原来的1行代码会转成多行。例如：

```
import torch
y = torch.transpose(image, 1, 0)
```


转换后：
```
import paddle
x = image
perm_0 = list(range(x.ndim))
perm_0[0] = 1
perm_0[1] = 0
y = paddle.transpose(x=x, perm=perm_0)
```

这是由于两者API的用法差异，无法通过一行代码来完成，需要增加若干辅助行来实现相同功能。


# 贡献代码

欢迎你向我们贡献代码。

根据API转换关系，我们将API分为三大类：

1. 一致的API：要求API功能一致，且API参数一致（如果Pytorch只比Paddle多out/dtype/device/layout/requires_grad/memory_format/inplace/generator/pin_memory参数，则也视作一致），这种只需增加json配置即可，最为容易

2. 不一致但可转换的API：包含 **Pytorch参数更多、参数不一致、API功能不一致、组合实现** 这几种情况，这种需要开发AST转换策略，难度较大

3. 不一致且无法转换的API：这种无法转换，可提供API映射关系，方便手动转换，见[Pytorch-Paddle API映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/pytorch_api_mapping_cn.html#pytorch-1-8-paddle-2-0-api) 

### 1. 一致的API

要求API的功能完全一致，且API参数的功能也完全一致。（仅名称不同时视作一致）

直接增加 paconvert/api_mapping.json 中的API配置，例如：

```python
"torch.nn.AvgPool2d": {
    "Matcher" : "GenericMatcher",
    "paddle_api": "paddle.nn.AvgPool2D",
    "args_list" : [
        "kernel_size", 
        "stride", 
        "padding", 
        "count_include_pad", 
        "ceil_mode", 
        "divisor_override"
    ],
    "kwargs_change": {
        "count_include_pad": "exclusive"
    }
}
```

- `Matcher`    :全部填写 `GenericMatcher`
- `paddle_api` :对应的Paddle API
- `args_list`  :按参数名称顺序填写
- `kwargs_change` :名称不同的参数对应关系，名称相同无需填写


### 2. 不一致但可转换的API

包含 **Pytorch参数更多、参数不一致、API功能不一致、组合实现** 这几种情况，需要开发基于AST的转换策略，有一定开发难度。

首先要在 paconvert/api_matcher.py 增加该 **APIMatcher** ，重写 `generate_code` 函数 ，以`torch.transpose`为例：

```

class TransposeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLACE = textwrap.dedent(
            '''
            {} = list(range(len({}.shape)))
            {}[{}] = {}
            {}[{}] = {}
            paddle.transpose({}, {})
            '''
        )
        perm = unique_name('perm')
        code = API_TEMPLACE.format(perm, kwargs['input'], 
                perm, kwargs['dim0'], kwargs['dim1'], 
                perm, kwargs['dim1'], kwargs['dim0'], 
                kwargs['input'], perm)
        return code
```

然后在 paconvert/api_mapping.json 中增加 json配置：

```
"torch.transpose" : {
    "Matcher": "TransposeMatcher",
    "args_list" : [
        "input",
        "dim0", 
        "dim1"
    ]
}
```

在本地开发中，为快速调试，可直接通过以下方式运行代码，无需反复安装：

```
python paconvert/main.py  --in_dir paconvert/tests/test_model.py  --out_dir paconvert/tests/out
```

