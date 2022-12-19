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

这是由于两者对应API的用法有差异，因此无法通过一行完成，将增加若干行Paddle API实现相同功能。


所有的API转换依据 [Pytorch-Paddle API映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/pytorch_api_mapping_cn.html#pytorch-1-8-paddle-2-0-api)

转换完成后，将打印总数、成功、失败的转换结果，对于无法转换的Pytorch API，会通过 >>> 在行前进行标识，需要进行手动修改并删除标记才可运行。


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

以以下API调用为例：
```
import torch

torch.reshape(torch.add(torch.abs(m), n), [3])

torch.reshape(torch.add(m.abs(), n), [3])

torch.reshape(torch.abs(m).add(n), [3])

torch.add(torch.abs(m), n).reshape([3])

torch.abs(m).add(n).reshape([3])

torch.add(m.abs(), n).reshape([3])

torch.reshape(m.abs().add(n), [3])

m.abs().add(n).reshape([3])
```

转换完成后：
```
import paddle
paddle.reshape(x=paddle.add(x=paddle.abs(x=m), y=n), shape=[3])
paddle.reshape(x=paddle.add(x=m.abs(), y=n), shape=[3])
paddle.reshape(x=paddle.abs(x=m).add(y=n), shape=[3])
paddle.add(x=paddle.abs(x=m), y=n).reshape(shape=[3])
paddle.abs(x=m).add(y=n).reshape(shape=[3])
paddle.add(x=m.abs(), y=n).reshape(shape=[3])
paddle.reshape(x=m.abs().add(y=n), shape=[3])
m.abs().add(y=n).reshape(shape=[3])
```

打印信息如下：

```txt
======================================
Convert Summary:
======================================
There are 24 Pytorch APIs in this Project:
 24  Pytorch APIs have been converted to Paddle successfully!
 0  Pytorch APIs are converted failed!
 Convert Rate is: 100.00%

Thank you to use Paddle Convert tool. You can make any suggestions to us.
```

一共有24个torch API被转换，因为上述每行 `torch.reshape(torch.add(torch.abs(m), n), [3])` 是包含连续3层的API调用，计作3个API。

转换完成后，将补全参数关键字信息、移除注释、多余空格。因为语法树转换为源码时，将采用标准写法来生成代码，这会使得与原来行数有一些差异。


# 贡献代码

根据API转换关系，我们将API分为两大类：
- 一致的API：要求API功能一致，且API参数一致（如果Pytorch较Paddle多out/dtype/device/layout/requires_grad/memory_format/inplace/generator/pin_memory参数，则也视作一致），通过一对一即可实现

- 不一致但可转换的API：包含Pytorch参数更多、参数不一致、API功能不一致、组合实现这几种情况，可以通过多行、多个API来实现，实现一对多的转换

- 不一致且无法转换的API：无法转换

欢迎你向我们贡献代码，对于 **一致的API** 目前通过json管理，你可以修改 paddleconverter/api_mapping.json，并补充以下信息：

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


对于 **不一致的API**，需要在 paddleconverter/api_matcher.py 中逐个增加 **Matcher** ，并重写 `generate_code` 函数 ，以`torch.transpose`为例：

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

然后在 paddleconverter/api_mapping.json 中增加json配置：

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
