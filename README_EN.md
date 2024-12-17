# overview
PaConvert is an API code conversion tool whose function is to convert Python project code into PaddlePaddle project code.

The principle is to use Python syntax tree analysis to generate the source code of the original PyTorch project as an abstract syntax tree, traverse, parse, match and edit it, and then get the abstract syntax tree of Paddle, and then convert it to the source code of Paddle project.

The conversion logic is static code scanning. Keep the style and structure of the original code unchanged, only convert the Python API, while other Python codes remain unchanged.

Note that only Python native APIs can be converted. For other third-party libraries (such as mmdet, mmcv, etc.) encapsulated based on the Pytorch API cannot be converted, and these APIs rely on manual conversion. It is recommended to copy this part of code and then use tools to convert it.

The conversion adopts a non-inplace method, convert the original project files one by one to the folder specified by `out_dir`. And the original file will not be changed to facilitate comparison and debugging before and after:

- Python files: recognize and convert Python APIs one by one
- requirements.txt: replace the torch installation dependency with paddlepaddle-gpu
- other documents: copy as is

For the Pytorch API, convert one-to-one as much as possible. However, in some cases, it is necessary to implement a Pytorch API with the help of multiple lines of Paddle code, which will lead to changes in the number of lines of code before and after conversion. For example:

```
import torch
y = torch.transpose(x, 1, 0)
```

After conversion：
```
import paddle
perm_0 = list(range(len(x.shape)))
perm_0[1] = 0
perm_0[0] = 1
y = paddle.transpose(x, perm_0)
```

This is because of the differences in the usage of the two APIs, which cannot be completed in one line. You must add several lines to achieve the same function.

All API conversions are based on [Pytorch-Paddle API Mapping table](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/pytorch_api_mapping_cn.html#pytorch-1-8-paddle-2-0-api).

The mapping table currently only contains the native Python API of 'torch.*'. For other third-party libraries encapsulated based on the Pytorch API (such as mmdet, mmcv, torchvision, etc.), the mapping relationship is not maintained, and depends on manual conversion.

After the conversion is completed, the statistical results of `total number of Pytorch APIs, number of conversion successes and conversion failures` will be printed to the terminal. For the Pytorch APIs that cannot be converted, we will mark them in front of the code line through the ">>>". You need to manually convert and delete the mark.


# Installation and Usage

Due to the use of some newer Python syntax tree features, an interpreter with >= python 3.8 is required.

1. Installation with pip

```bash
python3.8 -m pip install -U paconvert
paconvert --in_dir torch_project --out_dir paddle_project [--log_dir log_dir] [--log_level level] [--run_check] [--format]
```

2. Installation with source code

```bash
git clone https://github.com/PaddlePaddle/PaConvert.git
python3.8 paconvert/main.py --in_dir torch_project --out_dir paddle_project [--exclude_dirs exclude_dirs] [--log_dir log_dir] [--log_level level] [--run_check] [--no-format]
```

**Parameters**

```
Parameters:
--in_dir Enter the torch project file, either as a single file or as a folder
--out_dir Output paddle project file, same type as input, same as file or folder
--exclude_dirs  Optional, exclude converted files or folders, separate multiple items with a comma `,`
--log_dir Optional, the path to the output log, by default convert.log will be created in the current directory
--log_level Optional "INFO" "DEBUG", print log level, default "INFO"
--run_check Optional, tool self-test
--no-format Optional, disable format the converted code, default is False
```


# Example

Take the following API as an example：
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

After the conversion is completed：
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

Print the information as follows：

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

During the conversion process, a log will be printed in the terminal, which will record the number of APIs, files and lines that were not successfully converted, and the log file will be saved in `--log_dir`.

There are 8 torch APIs in this file, 7 of which were successfully converted, resulting in a conversion rate of 85.7%. If there is more than one file in the project, the cumulative data from all .py files will be counted.

For the successfully converted APIs, **the full API name, and parameter keywords will be filled in, the comments and extra blank lines will be removed**.Because when the syntax tree is reconverted to source code, the standard writing style is used to generate the code, and the code such as comments and blank lines cannot be recognized by the syntax tree and will be removed. Therefore there will be some differences in the number of lines before and after conversion.

For APIs that fail to convert, the **the full name of the torch API will be filled in**, and also the lines will be preceded by `>>>`. The user must manually convert the torch API, which can be found in the [Pytorch-Paddle API mapping Table](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/pytorch_api_mapping_cn.html#pytorch-1-8-paddle-2-0-api), then delete the marking.


# Contribute code

You are welcome to contribute code to us.

According to the API conversion relationship, we divide the API into three major categories：
- Consistent APIs: Requires consistent API functionality and consistent API parameters (if Pytorch only has more out/dtype/device/layout/requires_grad/memory_format/inplace/generator/pin_memory parameters than Paddle, then it is also considered consistent) and can be converted by one-to-one

- Inconsistent but convertible APIs：Contains more Pytorch parameters, inconsistent parameters, inconsistent API functionality, and combined implementations which may require one-to-many conversions via multiple lines and multiple APIs

- Inconsistent and non-convertible APIs：Unable to convert

#### 1. Consistent APIs

Only need to modify paconvert/api_mapping.json, add the following information：

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

- `Matcher` :For consistent APIs, fill in all `GenericMatcher`
- `paddle_api` :Corresponding Paddle API
- `args_list` :All torch api parameter names in order
- `kwargs_change` :Correspondence of parameter names (Note: When the function of the parameter is the same but the name is not the same, it is also regarded as the same)


#### 2. Inconsistent but convertible APIs

First you need to add **Matcher** in paconvert/api_matcher.py one by one, and override the `generate_code` function, using `torch.transpose` as an example:

```

class TransposeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLATE = textwrap.dedent(
            '''
            {} = list(range(len({}.shape)))
            {}[{}] = {}
            {}[{}] = {}
            paddle.transpose({}, {})
            '''
        )
        perm = unique_name('perm')
        code = API_TEMPLATE.format(perm, kwargs['input'],
                perm, kwargs['dim0'], kwargs['dim1'],
                perm, kwargs['dim1'], kwargs['dim0'],
                kwargs['input'], perm)
        return code
```

Then add the json configuration to paconvert/api_mapping.json：

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

Then `torch.transpose` will be transformed by the above one-to-many line method.

In local development, for quick debugging, you can run the code directly through the following way without repeated installation：

```
python3.8 paconvert/main.py  --in_dir tests/test_model.py  --out_dir tests/out
```
