# 概述
PaConvert是一个代码转换工具，能自动将其它深度学习框架训练或推理的**代码**，转换为PaddlePaddle的**代码**，方便**代码迁移**。

目前支持自动转换Pytorch代码，其它深度学习框架的支持后续新增中，其原理是通过Python AST语法树分析，将输入代码生成为抽象语法树，对其进行解析、遍历、匹配、编辑、替换、插入等各种操作，然后得到基于PaddlePaddle的抽象语法树，最后生成Paddle的代码。

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
--log_dir       可选，输出日志的路径，如不指定，则默认会在终端上打印日志
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
Start convert /workspace/test_code.py --> /workspace/PaConvert/paddle_project/test_code.py
[test_code.py:1] remove 'import torch' 
[test_code.py:2] remove 'import torch.nn as nn' 
[test_code.py:3] remove 'import torch.optim as optim' 
[test_code.py:4] remove 'import torch.nn.Linear as Linear' 
[test_code.py:5] remove 'import torch.nn.functional as F' 
[test_code.py] add 'import paddle' in first line
[test_code.py:25] [Not Support] convert torch.optim.SGD to Paddle is not supported currently
[test_code.py:26] [Not Support] convert torch.optim.lr_scheduler.MultiStepLR to Paddle is not supported currently
Finish convert /workspace/test_code.py --> /workspace/PaConvert/paddle_project/test_code.py


========================================
Convert Summary:
========================================
There are 10 Pytorch APIs in this Project:
 8  Pytorch APIs have been converted to Paddle successfully!
 2  Pytorch APIs are not supported to convert to Paddle currently!
 Convert Rate is: 80.000%

For these 2 Pytorch APIs that do not support to Convert now, which have been marked by >>> before the line, 
please refer to https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html 
and convert it by yourself manually. In addition, these APIs will be supported in future.

Thank you to use Paddle Code Convert Tool. You can make any suggestions to us.

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

首先你需要熟悉我们的[Pytorch-Paddle API映射关系](https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/pytorch_api_mapping_format_cn.md) 逻辑，映射关系是开发自动转换功能的前提条件。根据映射关系的复杂程度，我们将API分为7大类：

* 第1类为 `API可直接映射` ，此类情形最为容易。其又分为五种子情况：`无参数`、`参数完全一致`、`仅参数名不一致`、`仅 paddle 参数更多`、`仅参数默认值不一致`。

* 第2类为 `torch 参数更多`。如果 torch 和 paddle 都支持更多参数，统一写成`torch 参数更多`。

* 第3类为 `参数用法不一致`。比如 所支持的参数类型不一致(是否可以是元组)、参数含义不一致、返回参数类型不同。

* 第4类为 `组合替代实现` ，表示该 API 可以通过多个 API 组合实现。

* 第5类为 `用法不同：涉及上下文修改` ，表示涉及到上下文分析，需要修改其他位置的代码。

* 第6类为 `对应 API 不在主框架` 。例如 `torch.hamming_window` 对应 API 在 `paddlenlp` 中。

* 第7类为 `功能缺失` ，表示当前无该API功能，则不支持自动转换。

其中第1~6类API可按后续步骤开发，第7类需要先开发框架对应功能，目前不能开发自动转换功能。

我们需要定义一个概念：`Matcher` ：转换器，表示该API的转换规则。每一个API均对应一个转换规则。框架中已封装 `BaseMatcher` 作为转换器的基类。

上述的第1~6类的开发难度逐渐递增。目前框架里封装了 `GenericMatcher`：通用转换器， 可支持第1类API、大多数第2类API（仅多`layout` `memory_format`、`inplace`、`generator`、`non_blocking`、`pin_memory`、`dtype`、`requires_grad`、`device` 参数时）的转换。其他类API均需要自行编写 `Matcher`。

具体的开发步骤如下：

### 步骤1：配置JSON

在 `paconvert/api_mapping.json` 中增加该 API 的各项配置，每个配置字段的定义如下：

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
    },
    "paddle_default_kwargs": {}
}
```

```python
`Matcher`       :必须，转换器，执行转换时的核心转换规则。
`paddle_api`    :可选，对应的 Paddle API，仅 `GenericMatcher` 时需要。
`args_list`     :必须，根据顺序填写 torch api 的 **全部参数名**。
`kwargs_change` :可选，参数名称的差异，仅 `GenericMatcher` 时需要，无参数差异时不用填。
`paddle_default_kwargs` :可选，当 paddle 参数更多 或者 参数默认值不一致 时，可以通过该配置，设置参数默认值。
```

参考 [API映射关系表](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 看其属于哪种情况。

对于以下情况，都可以通过封装好的统一转换器：`GenericMatcher` 来处理。首先需配置 `paddle_api` 、 `args_list` ，其他需配置内容有：

- 第1类 `API可直接映射` ：
    - 无参数：无需其他配置
    - 参数完全一致：无需其他配置
    - 仅参数名不一致：增加 `kwargs_change` 
    - 仅paddle参数更多：增加 `paddle_default_kwargs`
    - 仅参数默认值不一致：增加 `paddle_default_kwargs`

- 第2类 `torch参数更多` ：仅多 `layout` `memory_format`、`inplace`、`generator`、`non_blocking`、`pin_memory`、`dtype`、`requires_grad`、`device` 参数时，与第1类按相同方式处理。

以 `torch.permute` 为例，首先参照 [torch.permute差异分析](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.permute.md)，其属于 **仅参数名不一致** 的情况，符合上述情况。

```python
"torch.permute" : {
    "Matcher": "GenericMatcher",
    "paddle_api": "paddle.transpose",
    "args_list" : ["input", "dims"],
    "kwargs_change": {
        "input": "x",
        "dims": "perm"
    }
}
```

如果不属于上述情况，则需要开发 **自定义的Matcher**，命名标准为：`API名+Matcher`， 例如 `torch.add` 可命名为`TorchAddMatcher`，然后在json中配置 `"Matcher"` 和 `"args_list"` 字段，其他字段无需配置，并进入到下面步骤2。


### 步骤2：编写Matcher（转换规则）

该步骤有一定难度，需要对AST相关知识有一定了解。

需在 `paconvert/api_matcher.py` 中增加自定义的Matcher，继承自 `BaseMatcher` 基类，并根据 是否含可变参数、是否为类方法调用 重写不同的函数。如果API参数有 `*size、*args` 等用法，例如 `torch.empty(*size)`，则含可变参数，否则不含可变参数。

1) 不含可变参数的API，符合大多数API情形，需重写：
* `generate_code()`: 传入的是字符串字典形式的关键字参数，即kwargs，根据该字典，组装字符串形式的代码并返回。

2) 含可变参数的API，如果不是类方法调用，需重写：
* `get_paddle_nodes()`：普通方法调用。传入的是AST形式的位置参数和关键字参数，即args和kwargs，需针对AST语法进行处理，生成新的AST节点并返回。

3) 含可变参数的API，如果是类方法调用，需重写：
* `get_paddle_class_nodes()`：类方法调用。主要用来处理类成员函数，与get_paddle_nodes不同的地方在于传入了func，根据这个func可以找到完整的调用链，首先需使用 `self.parse_func(func)` ，解析外部的调用链，然后 `self.paddleClass` 会存储调用类方法的对象。如 `x.abs().add(y)`中，对于`add(y)` 调用来说，它调用类方法的对象为 `x.abs()` ，即 `self.paddleClass='x.abs()'` ，通过 `self.paddleClass` 生成新的AST节点并返回。

对于类方法调用的API，由于其识别有一定难度，我们根据以下原则进行开发：（待补充）

以 `torch.transpose` 为例，首先参照 [torch.transpose差异分析](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/ops/torch.transpose.md)，其属于 **参数用法不一致** 的情况，不符合 `GenericMatcher` 的适用范围。
，则需要根据转写示例，将其转写规则写入到自定义的 `TransposeMatcher` 中。

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

其对应的json配置为：

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

编写自定义 `Matcher` 时，可以参考一些较为规范的Matcher（补充中）：
* 传入参数既可以是可变参数，也可以是列表或元组，例如 `TensorExpandMatcher`。

通过以下命令在本地调试：

```
python paconvert/main.py  --in_dir paconvert/test_code.py --log_level "DEBUG" --show_unsupport True`
```

### 开发规范

1. 编写的API转换器，必须考虑所有情形下的用户用法，涉及到多个参数的，应包含各种组合情况，不允许只考虑最简单的用户case。对任意case只允许有两种情况：a)正常转换且结果一致；b)不支持转换，此时返回None即可。不允许出现其他的错误情况，包括不限于 报错推出、错误转换 等问题。

以 `torch.Tensor.new_zeros` 为例，其至少包含以下12种可能用法：

```
case 1: x.new_zeros(2)
case 2: x.new_zeros(2, 3)
case 3: x.new_zeros([2, 3])
case 4: x.new_zeros((2, 3))

case 5:
shape = (2, 3)
x.new_zeros(shape, requires_grad=True)

case 6:
shape = (2, 3)
x.new_zeros(*shape, requires_grad=True, dtype=torch.float32, pin_memory=False)

case 7:
requires_grad_v = True
x.new_zeros(*shape, requires_grad=requires_grad_v, dtype=torch.float32, pin_memory=True)

case 8:
x.new_zeros(*shape, requires_grad=not True, dtype=torch.float32, pin_memory=False)

case 9:
pin_memory_v = True
x.new_zeros(*shape, requires_grad=False, pin_memory=pin_memory_v)

case 10:
x.new_zeros(2, 3, requires_grad=True, pin_memory=False)

case 11:
x.new_zeros(*x.size())

case 12:
x.new_zeros(x.size())
```

2. 在穷举所有可能的用法case后，将其全部加入到单测中，并对比结果一致，要求至少编写5种case（越多约好）。单测写法为：（待补充）

3. 需要使用验证集通过，流水线xxx通过即可。

4. 由于考虑的场景仍可能不全面，可能引入非常隐蔽的case bug，开发者需要对自身开发的API转换规则负责后续维护，并解决考虑不全面的问题。

总的来说，Matcher转换规则的开发具有一定的挑战性，是一项非常细心以及考验思考广度的工作。

