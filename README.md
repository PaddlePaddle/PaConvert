# 概述
Paddleconverter是一款工具，其功能是将Pytorch项目训练代码从转换为PaddlePaddle训练代码。

其原理是借助Python语法树分析，将原PyTorch脚本生成为抽象语法树，对其进行遍历、解析、编辑，然后替换为Paddle的抽象语法树，再转回为Paddle脚本。

转换逻辑为静态代码扫描，保持原代码的风格与结构不变，只转换相应的Pytorch API，其他Python代码保持原样不变。

转换采用非inplace的方式，不修改原文件，将原Pytorch项目文件一一转换到 `out_dir` 指定的文件夹中：

- Python文件，逐个转换
- requirements.txt 转换其中的 torch 安装依赖
- 其他文件，原样拷贝

对于一个 Pytorch API，尽可能按一对一的形式转换，但在某些情形下，会借助多行Paddle代码来实现一个Pytorch API，这会导致转换前后的代码行数是不同的。例如：

```
import torch
y = torch.transpose(x, 1, 0)
```

转换后：
```
import paddle
perm_0 = range(len(x.shape))
perm_0[1] = 0
perm_0[0] = 1
y = paddle.transpose(x, perm_0)
```

这是由于两者API的用法差异，无法通过一行完成，必须增加若干行来实现相同功能。


所有的API转换是依据 [Pytorch-Paddle API映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/pytorch_api_mapping_cn.html#pytorch-1-8-paddle-2-0-api) 来进行的。

在转换完成后，Pytorch API总数、成功、失败数量统计将会打印到终端，对于无法转换的Pytorch API，我们会通过 >>> 在代码行前面进行标识，你需要手动修改并删除该标记。


# 安装

1. 使用pip安装

```bash
pip install -U paddleconverter-1.0-py3-none-any.whl 
paddleconverter --help # show paddleconverter help
paddleconverter --run_check 1 # tool run check
```

如果你的机器安装了多个Python，建议直接使用最新python解释器来进行转换，例如：

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
paddleconverter --in_dir torch_project --out_dir paddle_project --log_dir log_dir

参数：
--in_dir 输入torch项目文件夹，可以为单个文件或文件夹
--out_dir 输出paddle项目文件夹，必须为文件夹
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
======================================
Convert Summary:
======================================
There are 8 Pytorch APIs in this Project:
 7  Pytorch APIs have been converted to Paddle successfully!
 1  Pytorch APIs are converted failed!
 Convert Rate is: 85.70%

Thank you to use Paddle Convert tool. You can make any suggestions to us.
```

一共有8个torch API，其中7个被成功转换，转换率为85.7%。

成功转换的API，将 补全API全名、参数关键字、移除注释、移除多余空行。因为语法树转换为源码时，将采用标准写法来生成代码，这会使得与原来行数有一些差异。

对于未成功转换的API，将 补全torch API全名，同时在行前通过 `>>>` 的形式加以标注，用户必须对该torch API进行手动转换，并删除标注。


# 贡献代码

欢迎你向我们贡献代码。

根据API转换关系，我们将API分为三大类：
- 一致的API：要求API功能一致，且API参数一致（如果Pytorch较Paddle多out/dtype/device/layout/requires_grad/memory_format/inplace/generator/pin_memory参数，则也视作一致），通过一对一即可实现

- 不一致但可转换的API：包含Pytorch参数更多、参数不一致、API功能不一致、组合实现这几种情况，可以通过多行、多个API来实现，实现一对多的转换

- 不一致且无法转换的API：无法转换

## 1. **一致的API** 

仅需修改 paddleconverter/api_mapping.json，并补充以下信息：

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


## 2. **不一致的API**

首先需要在 paddleconverter/api_matcher.py 中逐个增加 **Matcher** ，并重写 `generate_code` 函数 ，以`torch.transpose`为例：

```

class TransposeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLACE = textwrap.dedent(
            '''
            {} = range(len({}.shape))
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

然后根据 paddleconverter/api_mapping.json 中增加 json配置：

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

可通过上述一对多行的方式进行转换。

在本地开发中，为快速调试，可直接通过以下方式运行代码：

```
python paddleconverter/main.py  --in_dir paddleconverter/tests/test_model.py  --out_dir paddleconverter/tests/out
```
