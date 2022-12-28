# 概述
paddleconverter是一款API代码转换工具，其功能是将Pytorch项目代码转换为PaddlePaddle项目代码。

其原理是借助Python语法树分析，将原PyTorch项目源码的生成为抽象语法树，对其进行遍历、解析、匹配、编辑，然后得到Paddle的抽象语法树，再转回为Paddle的项目源码。

转换逻辑为静态代码扫描，保持原代码的风格与结构不变，只转换相应的Pytorch API，其他Python代码保持原样不变。

转换采用非inplace的方式，将原Pytorch项目文件一一转换到 `out_dir` 指定的文件夹中，不修改原文件，方便前后对比：

- Python文件：逐个识别torch API并转换
- requirements.txt： 转换其中的 torch 安装依赖
- 其他文件：原样拷贝

对一个 Pytorch API，尽可能按一对一的形式转换，但在某些情形下，必须借助多行Paddle代码才能实现一个Pytorch API，这会导致转换前后的代码行数改变。

例如：

```
import torch
y = torch.transpose(x, 1, 0)
```

转换后：
```
import paddle
perm_0 = list(range(len(x.shape)))
perm_0[1] = 0
perm_0[0] = 1
y = paddle.transpose(x, perm_0)
```

这是由于两者API的用法差异，无法通过一行完成，必须增加若干行来实现相同功能。

所有的API转换是依据 [Pytorch-Paddle API映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/pytorch_api_mapping_cn.html#pytorch-1-8-paddle-2-0-api) 来进行的。注意：该映射表仅针对torch.* 的API，也就是只有此部分API在待转换范围里。对于其他的库，例如`torchvision`，由于并没有映射表关系，均不在待转换范围里。

在转换完成后，`Pytorch API总数、转换成功数、转换失败数` 的统计结果将会打印到终端，对于无法转换的Pytorch API，我们会通过 `>>>` 在代码行前面进行标识，你需要手动转换并删除该标记。


# 安装

1. 使用pip安装

```bash
pip install -U paddleconverter-1.0-py3-none-any.whl 
paddleconverter --help # show paddleconverter help
paddleconverter --run_check 1 # tool run check
```

如果你的机器安装了多个Python，建议使用最新的python解释器来进行转换，以使用最新的ast功能，例如：

```bash
python3.9 -m pip install -U paddleconverter-1.0-py3-none-any.whl 
```

2. 使用源码安装

```bash
git clone https://github.com/zhouwei25/paddleconverter.git
cd paddleconverter
python setup.py bdist_wheel
cd ..
python -m pip install -U paddleconverter/dist/*.whl
```

# 用法

```bash
paddleconverter --in_dir torch_project --out_dir paddle_project [--log_dir log_dir]

参数：
--in_dir 输入torch项目文件，可以为单个文件或文件夹
--out_dir 输出paddle项目文件，输入文件，则也应为文件，输入文件夹，则也应为文件夹
--log-dir 可选，输出日志的路径，默认值不生成输出日志文件

```

# 简单示例

以下API为例：
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
```

转换完成后：
```
""" This file has been converted by Paddle converter, thanks to use, you can remove this mark"""
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
>>> sgd = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
```

打印信息如下：

```txt
=======================================
PyTorch to Paddle Convert Start ----->:
=======================================
Start convert /workspace/paddleconverter/paddleconverter/tests/temp_code.py --> /workspace/paddleconverter/tests/temp_out/temp_code.py
[temp_code.py:1] remove 'import torch' 
[temp_code.py:2] remove 'import torch.nn as nn' 
[temp_code.py:3] remove 'import torch.optim as optim' 
[temp_code.py:4] remove 'import torch.nn.Linear as Linear' 
[temp_code.py:5] remove 'import torch.nn.functional as F' 
[temp_code.py] add 'import paddle' in first line
[temp_code.py:1] [Success]convert torch.nn.Module to Paddle
[temp_code.py:11] [Success]convert torch.nn.Linear to Paddle 
[temp_code.py:12] [Success]convert torch.nn.Linear to Paddle 
[temp_code.py:13] [Success]convert torch.nn.Linear to Paddle 
[temp_code.py:20] [Success]convert torch.add to Paddle 
[temp_code.py:21] [Success]convert torch.nn.functional.relu to Paddle 
[temp_code.py:15] [Success]convert torch.no_grad to Paddle 
[temp_code.py:24] [Failed]can not convert torch.optim.SGD to Paddle 
[temp_code.py] Mark this file which has been converted already
Finish convert /workspace/paddleconverter/paddleconverter/tests/temp_code.py --> /workspace/paddleconverter/tests/temp_out/temp_code.py

======================================
Convert Summary:
======================================
There are 8 Pytorch APIs in this Project:
 7  Pytorch APIs have been converted to Paddle successfully!
 1  Pytorch APIs are converted failed!
 Convert Rate is: 87.50%

For these 1 failed converted Pytorch APIs, Please refer to https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/pytorch_api_mapping_cn.html#pytorch-1-8-paddle-2-0-api and modify it by yourself manually!

Thank you to use Paddle Convert tool. You can make any suggestions to us.

```

一共有8个torch API，其中7个被成功转换，该文件的转换率为85.7%，如果项目中有多个文件，会统计所有.py文件累计的数据。

对于转换成功的API，将 **补全API全名、参数关键字、移除注释、移除多余空行**。因为语法树重新转换为源码时，会采用标准写法来生成代码，而注释、空行等代码无法被语法树识别，将被移除。因此转换前后行数会有一些差异。

对于转换失败的API，将 **补全为torch API全名**，同时在行前通过 `>>>` 的形式加以标注，用户必须对该torch API进行手动转换，可参考[Pytorch-Paddle API映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/pytorch_api_mapping_cn.html#pytorch-1-8-paddle-2-0-api)，然后删除标注。


# 贡献代码

欢迎你向我们贡献代码。

根据API转换关系，我们将API分为三大类：
- 一致的API：要求API功能一致，且API参数一致（如果Pytorch只比Paddle多out/dtype/device/layout/requires_grad/memory_format/inplace/generator/pin_memory参数，则也视作一致），通过一对一即可转换

- 不一致但可转换的API：包含Pytorch参数更多、参数不一致、API功能不一致、组合实现这几种情况，可能要通过多行、多个API来进行一对多的转换

- 不一致且无法转换的API：无法转换

#### 1. 一致的API

仅需修改 paddleconverter/api_mapping.json，补充以下信息：

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

- `Matcher` :对于一致的API，全部填写 `GenericMatcher`
- `paddle_api` :对应的Paddle API
- `args_list` :参数名，按参数名称顺序填写
- `kwargs_change` :参数名的对应关系（注: 参数功能一致仅名字不一致时也视作一致）


#### 2. 不一致但可转换的API

首先需要在 paddleconverter/api_matcher.py 中逐个增加 **Matcher** ，并重写 `generate_code` 函数 ，以`torch.transpose`为例：

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

然后在 paddleconverter/api_mapping.json 中增加 json配置：

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

则 `torch.transpose` 将通过上述一对多行的方式进行转换。

在本地开发中，为快速调试，可直接通过以下方式运行代码，无需反复安装：

```
python paddleconverter/main.py  --in_dir paddleconverter/tests/test_model.py  --out_dir paddleconverter/tests/out
```

