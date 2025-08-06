# 如何贡献代码

## 步骤1: 开发环境准备与CI检查
### 依赖项
在开发本项目之前，请确保已经安装了以下依赖项：

#### 最新版本的paddle库和torch库
```bash
# cpu 版本的paddle
python -m pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html
```

```bash
# cpu 版本的torch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 其它依赖库
```bash
pip install -r tests/requirements.txt
```

### 代码审查
代码审查包括两部分，一个是基于现有开源代码格式规范的审查，通过pre-commit来审查，另一个是基于自定义要求的审查，在tools下。

#### Pre-commit审查

本项目使用了 pre-commit 工具来进行代码规范和格式化，主要包括black,flake8,pylint，在CI阶段代码审查执行，具体规范配置请见根目录.pre-commit-config.yaml文件。在提交代码之前，pre-commit 钩子将自动运行相关的检查和修复。本地运行方式如下

```bash
# 要安装 pre-commit 钩子，请执行以下命令：
pip install pre-commit
pre-commit install
# 调整代码格式和规范错误
pre-commit run --file [file_name]
```
#### 自定义审查 (如果无需本地调试，请跳过)

本项目使用了基于自定义要求的审查，相关代码文件在tools和scripts下,可自行本地调试。

相关CI的测试代码和本地scriptes的脚本文件对应如下
|  CI名称   | repo对应脚本文件  |
|  ----  | ----  |
| PR-CI-ModelTest   | scripts/modeltest_check.sh |
| PR-CI-CodeCosistency   | scripts/consistency_check.sh |
| PR-CI-CodeStyle   | scripts/code_style_check.sh |
| PR-CI-UnitTest   | scripts/unittest_check.sh |
| PR-CI-Coverage   | scripts/code_coverage_check.sh |
| PR-CI-Pipeline   | scripts/install_check.sh |
| PR-CI-PRTemplate   | scripts/PRtemplate_check.sh |
```

运行对应CI文件需修改scriptes中*.sh的环境变量DEVELOP_IF="ON".

```bash
本地单个CI测试方法
bash scripts/modeltest_check.sh
bash scripts/consistency_check.sh
bash scripts/code_style_check.sh
bash scripts/unittest_check.sh
bash scripts/code_coverage_check.sh
bash scripts/install_check.sh
bash scripts/PRtemplate_check.sh

本地全部CI测试方法
bash scripts/run_ci.sh
```

### 合入规范

合入**必须**要求通过全部CI检测，原则上禁止强行Merge，如果有代码风格阻塞，可以讨论是否禁止某一条pre-commit规范，**必须**要求一个Reviewer的approve，禁止出现敏感代码。

提交PR时，请尽可能按照以下规范
```bash
### PR Docs
<!-- Describe the docs PR corresponding the APIs -->
https://github.com/PaddlePaddle/docs/pull/_prID
### PR APIs
<!-- APIs what you’ve done -->
torch.transpose
torch.Tensor._index_copy
torch.permute
...
```

## 步骤2：编写API映射关系

首先你需要熟悉我们的 [Pytorch-Paddle API映射关系表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html#pytorch-1-13-paddle-2-4-api) ，映射关系相当于人工转换的思路，其是开发自动转换功能的前提，基于这些映射关系，我们才可以进行后续的开发。

根据映射关系的复杂程度，我们将API分为7大类：

* 第1类为 `API可直接映射` ，此类情形最为容易。其又分为五种子情况：`无参数`、`参数完全一致`、`仅参数名不一致`、`仅 paddle 参数更多`、`仅参数默认值不一致`。

* 第2类为 `torch 参数更多`。如果 torch 和 paddle 都支持更多参数，统一写成`torch 参数更多`。

* 第3类为 `参数用法不一致`。比如 所支持的参数类型不一致(是否可以是元组)、参数含义不一致、返回参数类型不同。

* 第4类为 `组合替代实现` ，表示该 API 可以通过多个 API 组合实现。

* 第5类为 `用法不同：涉及上下文修改` ，表示涉及到上下文分析，需要修改其他位置的代码。

* 第6类为 `对应 API 不在主框架` 。例如 `torch.hamming_window` 对应 API 在 `paddlenlp` 中。

* 第7类为 `功能缺失` ，表示当前无该API功能，则不支持自动转换。

其中第1~6类API需按后续步骤开发，第7类由于无法支持自动转换，仅在 API映射表页面 标注 **功能缺失** 即可，无需其他开发。。

对于一个待支持转换的Pytorch API，首先查阅 [Pytorch-Paddle API映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html)，如果已经有了该API的映射关系，则无需新增映射关系文档，可以直接参考来编写转换规则。但如果发现其问题，则需要进行修复。

> 注意：当前已有一部分存量映射关系文档，但可能存在错误或考虑不全面之处，在开发自动转换规则时，如发现文档问题，需要对这些文档进行校正修改。

如果没有该API映射关系，则需要自行分析API，并根据统一模板来编写映射关系文档，提交PR到 https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference 目录下。统一模板详见：[API映射关系模板](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md)。

## 步骤3：配置JSON

在 `paconvert/api_mapping.json` 中增加该 API 的各项配置，每个配置字段的定义如下：

```json
{
  "torch.permute" : {
    "Matcher": "GenericMatcher",
    "paddle_api": "paddle.transpose",
    "args_list" : ["input", "dims"],
    "kwargs_change": {
      "input": "x",
      "dims": "perm"
    }
  },
  "unsupport_args": [],
  "paddle_default_kwargs": {}
}
```

```python
Matcher       :必须，转换器，亦称为转换规则，表示执行转换时的核心逻辑。每一个API均对应一种转换规则，所有API都需要配置。
paddle_api    :可选，对应的 Paddle API，如果是 `GenericMatcher` ，则必须配置。
min_input_args:可选，至少输入的位置参数个数，如果是类方法API（例如torch.Tensor.*），则必须配置，可用来辅助判断类方法API
args_list     :必须，根据顺序填写 torch api 的 `全部参数名`，所有API都需要配置。
kwargs_change :可选，参数名称的差异，如果是 `GenericMatcher` 且有参数名称差异时，则必须配置。
unsupport_args:可选，不支持的参数功能，通过该字段配置后，这些参数如果被使用，将直接标记为不支持转换。
paddle_default_kwargs :可选，当 `paddle 参数更多` 或者 `参数默认值不一致` 时，可以通过该配置，设置参数默认值。
```

对于一个待开发API，首先依据步骤1的映射关系，确定其属于哪种分类情况。

对于以下映射关系的分类，都可以通过框架封装好的通用转换器：`GenericMatcher` 来处理：

- 第1类 `API可直接映射` ：
    - 无参数：无需其他配置
    - 参数完全一致：无需其他配置
    - 仅参数名不一致：增加 `kwargs_change` 配置
    - 仅paddle参数更多：增加 `paddle_default_kwargs` 配置
    - 仅参数默认值不一致：增加 `paddle_default_kwargs` 配置

- 第2类 `torch参数更多` ：仅多 `layout` `memory_format`、`inplace`、`generator`、`non_blocking`、`pin_memory`、`dtype`、`requires_grad`、`device` 参数时，与第1类按相同方式处理。

除了以上字段以及必选字段 `Matcher` 和 `args_list` 外，`GenericMatcher` 还需配置 `paddle_api` 。

以 `torch.permute` 为例，首先参照 [torch.permute映射关系](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/torch/torch.permute.md)，其属于 **仅参数名不一致** 的情况，符合上述类型。因此可通过`GenericMatcher`来实现，在json中配置必选的 `Matcher` 和 `args_list` 字段，还需配置 `paddle_api` 和 `kwargs_change`。

```json
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

如果不属于上述情形，则需要 **新增Matcher**，当前已经有100+种其他`Matcher`，建议尽可能复用已有`Matcher`，提升代码的可复用性。如果已有的100+种 `Matcher` 无法满足要求，才需要新增开发 `Matcher`。

新增Matcher的命名标准为：`API名+Matcher` 。例如 `torch.transpose` 可命名为`TransposeMatcher` ，`torch.Tensor.transpose` 可命名为 `TensorTransposeMatcher`。详见下面步骤。

## 步骤4：编写Matcher（转换规则）

该步骤有一定难度，需要对AST相关知识有一定熟悉。

首先在 `paconvert/api_matcher.py` 中增加自定义的Matcher，继承自 `BaseMatcher` 基类，然后在json中配置必选的 `Matcher` 字段 和 `args_list(类方法无需配置)` 字段，其他字段不做要求，可选配 `paddle_api` 字段。

根据 **是否为类方法** ，共有三种开发方式：

### 方式一：适用非类方法

判断标准：**所有不是类方法的API**。

根据 **是否支持可变参数** ，又分为以下两种情况：

> 可变参数是指Python语法中的`*args` 用法，例如 `torch.empty(*size)`，则含可变参数。通常来讲，大多数API不含可变参数。

**1）支持可变参数**，则重写：

* `get_paddle_nodes(self, args, kwargs)`: 传入的是AST形式的位置参数和关键字参数（其中args为ast.Node的列表，kwargs为ast.keyword的列表），需针对AST语法进行处理，组装代码并生成新的AST节点返回。

以 `torch.chain_matmul` 为例，由于其支持可变参数，可以传入任意个矩阵，因此重新 `get_paddle_nodes` 函数，通过`parse_args`、`parse_kwargs`来解析AST形式的参数为字符串，具体代码如下：

```python
class Chain_MatmulMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        new_args = self.parse_args(args)
        new_kwargs = self.parse_kwargs(kwargs)

        code = "{}".format(new_args[0])
        for arg in new_args[1:]:
            code = code + " @ {}".format(arg)
        if "out" in new_kwargs and new_kwargs["out"] is not None:
            code = "paddle.assign({}, output={})".format(code, new_kwargs["out"])

        return ast.parse(code).body
```

对应的json配置为：

```json
"torch.chain_matmul": {
    "Matcher": "Chain_MatmulMatcher",
}
```

**2）不支持可变参数**，则重写：

* `generate_code(self, kwargs)`: 传入的kwargs是 `字符串字典` 形式的关键字参数，根据kwargs组装字符串形式的代码并返回。其相比 `get_paddle_nodes` 更为high_level一些，无需自行处理AST节点与字符串的各种解析、转换，直接处理纯字符串组装代码即可，相对容易些。

以 `torch.transpose` 为例，首先参照 [torch.transpose映射关系](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/torch/torch.transpose.md)，其属于 **参数不一致** 的情况，不符合 `GenericMatcher` 的适用范围。因此需要编写自定义Matcher：`TransposeMatcher` 。

由于 `torch.transpose` 不支持可变参数，因此重新 `generate_code` 函数，具体代码如下：

```python
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

对应的json配置为：

```json
"torch.transpose" : {
    "Matcher": "TransposeMatcher",
    "args_list" : [
        "input",
        "dim0",
        "dim1"
    ]
}
```

### 方式二：适用于类方法且可以识别

由于 **类方法** 可能与其他Python class的类方式混淆，导致无法识别，如果该API具有独特的深度学习API名称，例如 `x.backward()` 、 `sgd.step()` ，则可以准确识别，可使用本方式开发。

判断标准：**类方法API且具有独特的API名**。需要与 numpy、scipy、python原生class(list/tuple/set/dict等)的类方法进行对比，若有任意相同API，则不符合此标准。

根据 **是否支持可变参数** ，又分为以下两种情况：

**1）支持可变参数**，则重写：

* `get_paddle_class_nodes(self, func, args, kwargs)`: 主要用来处理类成员函数，与 `get_paddle_nodes` 不同的地方在于传入了func，根据这个func可以找到完整的调用链，首先需使用 `self.parse_func(func)` ，解析外部的调用链，然后 `self.paddleClass` 会存储调用类方法的对象。如 `x.abs().add(y)`中，对于`add(y)` 调用来说，它调用类方法的对象为 `x.abs()` ，即 `self.paddleClass='x.abs()'` ，通过 `self.paddleClass` 来组装新的调用代码，并生成新的AST节点返回。


**2）不支持可变参数**，则重写：

* `generate_code(self, kwargs)`: 传入的是字符串字典形式的关键字参数，即kwargs，根据该字典，组装字符串形式的代码并返回。其相比 `get_paddle_class_nodes` 更为high_level一些，已经进行了 `self.paddleClass`的设置，也无需自行处理AST节点与字符串的各种解析、转换，直接处理纯字符串组装代码即可，相对容易些。

以 `torch.Tensor.repeat_interleave` 为例，由于该API名称很长，不容易出现误识别；numpy等其他库也没有 `ndarray.repeat_interleave` API，因此符合此标准。参照 [torch.transpose映射关系](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/torch/torch.transpose.md)，其属于 **仅参数名不一致** 的情况，符合 `GenericMatcher` 的适用范围。因此只需配置json字段即可，其中已编写了通用的 `generate_code` 函数。


```json
"torch.Tensor.repeat_interleave": {
  "Matcher": "GenericMatcher",
  "paddle_api": "paddle.Tensor.repeat_interleave",
  "args_list": [
    "repeats",
    "dim",
    "output_size"
  ],
  "kwargs_change": {
    "dim": "axis"
  }
}
```

所有不符合方式二的类方法，均采用方式三开发。

### 方式三：适用于类方法但无法识别

方式三与方式二的区别在于，其原理为**保持转换前后代码不变，则可消除无法识别的问题**。同时增加后台辅助代码对API的调整，来保证代码在完全不变的前提下，仍可正常运行。根据是否需要辅助代码，其又分为 **不需要辅助代码** 、**需要辅助代码** 两种情况。

**1）不需要辅助代码**

判断标准：**代码保持完全不变，即可直接正常运行**。此时直接在json中配置已封装好的 `UnchangeMatcher` 即可，无需编写新的Matcher。

以 `torch.Tensor.tan` 为例：

```json
"torch.Tensor.tan": {
    "Matcher": "UnchangeMatcher"
}
```

基于[Pytorch-Paddle API映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html)，其中的 **无参数、参数完全一致、仅paddle 参数更多** 分类，均符合该情形。

**2）需要辅助代码**

对于某些API，如果 **代码保持完全不变时，无法直接运行**，我们就需要在后台通过辅助代码对Paddle相应类方法进行一些修改，使得在 **转换前后代码保持不变** 的前提下，仍可正常运行。

**开发方式**：在 `get_paddle_class_nodes` 或 `generate_code` 增加相应的判断：
- 对于 **不需要辅助代码** 即可运行的用法，直接返回 'unchange'
- 对于 **需要辅助代码** 才可运行的用法，首先要额外重写 `generate_utils_code` 函数，其是模仿Pytorch类API用法的辅助代码，然后显式的调用 `enable_utils_code` ，此时将在后台模块里注入辅助代码，最后再返回 'unchange' 即可

由于 **辅助代码** 会改变原Paddle Tensor API的用法，应可能取两者功能的并集，所以在编写辅助代码的函数时，需采用 `*args、**kwargs` 来描述参数，从而可同时兼容pytorch与原paddle的参数名，例如 `paddle.Tensor.add` 被辅助函数修改后应同时可支持输入 `y` 与 `other` 作为第二个输入。

另外我们需要判断用户的不同用法，在必要的情形下才 `enable_utils_code` 使用辅助函数，以尽可能的减少辅助代码的使用。

基于[Pytorch-Paddle API映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html)，我们可参考以下原则来判断是否需要辅助代码：

|分类|不需要辅助代码的用法|需要辅助代码的用法|
|---|---|---|
|对应API名称不一致||全部需要辅助代码|
|仅参数名不一致|未指定关键字参数|指定了关键字参数|
|torch参数更多|未使用torch多的参数|使用了torch多的参数|
|参数不一致|未使用不一致的用法|使用了不一致的用法|
|其他分类||全部需要辅助代码|

以 `torch.Tensor.reshape` 为例，其映射关系分类属于 **参数不一致**，是由于torch的shape既可为可变参数，也可为list/tuple，而Paddle仅支持list/tuple，因此我们只需对**可变参数**的用法 `enable_utils_code` 。

```python
class TensorReshapeMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def reshape(self, *args, **kwargs):
                if args:
                    if len(args)==1 and isinstance(args[0], (tuple, list)):
                        return paddle.reshape(self, args[0])
                    else:
                        return paddle.reshape(self, list(args))
                elif kwargs:
                    return paddle.reshape(self, **kwargs)

            setattr(paddle.Tensor, 'reshape', reshape)
            """
        )
        return CODE_TEMPLATE

    def get_paddle_class_nodes(self, func, args, kwargs):
        if len(args) == 1 and isinstance(args[0], (ast.List, ast.Tuple)):
            return "unchange"

        if len(kwargs) == 1 and "shape" in kwargs:
            return "unchange"

        self.enable_utils_code()
        return "unchange"

```

对应的json配置为：

```json
"torch.Tensor.reshape": {
    "Matcher": "TensorReshapeMatcher"
}
```

```python
转换前：
x.reshape(2, 3)

转换后：
############################## 相关utils函数，如下 ##############################

def reshape(self, *args, **kwargs):
    if args:
        if len(args)==1 and isinstance(args[0], (tuple, list)):
            return paddle.reshape(self, args[0])
        else:
            return paddle.reshape(self, list(args))
    elif kwargs:
        assert 'shape' in kwargs
        return paddle.reshape(self, shape=kwargs['shape'])

setattr(paddle.Tensor, 'reshape', reshape)

############################## 相关utils函数，如上 ##############################

x.reshape(2, 3)
```
如果使用 PaConvert 转换单个文件，则会在该文件头部自动添加辅助函数代码。
如果使用 PaConvert 转换整个目录，则会在目录下自动生成 `paddle_utils.py` 文件，辅助函数代码会写入该文件中。

### 开发规范

1) 代码精简与美观性。要求尽可能只通过一行代码、一个API来实现（代码越少越好）。如果确实无法实现，才考虑通过多行代码、多个API来辅助实现该功能。

2) 维护与负责。由于单测可能覆盖不全面，导致引入了非常隐蔽的用法bug，开发者需要后续维护自己开发的API转换规则。解决新反馈的用法case问题。

3) API功能缺失。如果是整个API都缺失的，只需在API映射表中标注 **功能缺失** 即可，无需其他开发。如果是API局部功能缺失，则对功能缺失点，在代码中返回None表示不支持，同时在API映射表中说明此功能点 **Paddle暂无转写方式**，同时编写单测但可以注释掉不运行；对其他功能点正常开发即可。

4) 别名实现。如果一个API是别名API(alias API)，例如 `torch.nn.modules.GroupNorm` 是 `torch.nn.GroupNorm` 的别名，那么就无需编写相关 Matcher，只需在 `paconvert/api_alias_mapping.json` 中增加该别名 API 的配置，同时也需要编写单测文件。

    ```bash
    {
      "torch.nn.modules.GroupNorm": "torch.nn.GroupNorm"
    }
    ```

### 开发技巧

1）可以参考一些写的较为规范的Matcher：
- 传入参数既可以是可变参数，也可以是列表或元组时，例如 `TensorExpandMatcher`
- (待补充)...

2）由于AST语法分析是静态代码分析，也就是Matcher被执行时的并未到代码的运行期，无法知道某个变量的运行值，要避免对变量运行值的判断，否则可能引入错误。

例如：如果在Matcher里的以下判断形式 `if 'True' == kwargs['pin_memory']` ，对以下Python代码将失效，因为 `kwargs['pin_memory']` 只有到代码运行期其值才为'True'，在AST语法分析期，只能获取 `kwargs['pin_memory']='temp'` ，无法获取具体运行值，所以上述判断将失效。

```python
temp = True
torch.tensor(1., pin_memory=temp)
```

因此在Matcher编写时，需格外注意此时为静态语法分析期，避免判断运行期的值；如必须判断，则不要在Matcher中判断，将其挪到转换后的代码也就是在运行期来判断。例如转成以下形式：

```python
temp = True
paddle.to_tensor(1., place=paddle.CUDAPinnedPlace() if temp else None)
```

3）谨慎通过多行代码来实现，多余代码行数将插入到该作用域中，最后一行将替换原本的API，注意不能破坏原代码的语法树结构。例如：

```python
if x:
    out = torch.add(torch.transpose(x, 1, 0).add(y), z)
```

其中 `torch.transpose(x, 1, 0)` 会通过5行代码实现:

```python
x = x
perm_0 = list(range(x.ndim))
perm_0[0] = 1
prem_0[1] = 0
paddle.transpose(x=x, perm=perm_0)
```

其中前4行将直接插入到该作用域中，第5行将替换原本的ast.Call: `torch.transpose(x, 1, 0)`，转换完该API的中间结果为：

```python
if x:
    x = x
    perm_0 = list(range(x.ndim))
    perm_0[0] = 1
    prem_0[1] = 0
    out = torch.add(paddle.transpose(x=x, perm=perm_0).add(y), z)
```

为避免破坏语法树结构，最后一行仅可为`ast.Call/ast.Name/ast.Constant/ast.Attribute/ast.Compare...`等较小的子节点形式，如果为`ast.Assign/ast.For...`等根节点形式，则容易破坏原来的语法树结构，当前会被自动过滤掉。

4）在开发时可能需要查询各种 `ast.Node` 的组成属性，可以参考 https://greentreesnakes.readthedocs.io/en/latest/nodes.html#function-and-class-definitions ，同时也建议熟悉各种常用AST节点，有利于开发效率的提升。


## 步骤5：编写单元测试

**单测写法**：

* **单测位置**：所有的单测文件均放在`tests`目录下，单测文件命名以`test_`为前缀，后面接测试的`API`名称（PyTorch API名称即可，保留大小写，无需Module前缀）。例如 `torch.nn.functional.relu` 命名为 `test_relu.py` ， `torch.Tensor.add` 命名为 `test_Tensor_add.py` 。

* **默认检查逻辑**：采用`pytest`作为单测框架。一般情况下，用户只需要在单测文件中调用 `APIBase` 类的 `run()` 方法，传入 `pytorch_code` 和需要判断的 `Tensor` 变量名列表即可，参考 [torch.permute测试用例](https://github.com/PaddlePaddle/PaConvert/tree/master/tests/test_permute.py)。 `run()` 方法会调用`compare()`函数，该方法默认检查逻辑为：转换前后两个`Tensor`的 `计算数值、数据类型、stop_gradient属性、形状` 是否一致。

* **关闭数值检查**：对于随机数API，允许 `计算数值` 不同，可通过设置 `run(check_value=False)` 来实现。其他API不允许关闭数值检查。

* **不支持的检查**：对于目前不支持的转换，负责转换的`Matcher` 需要返回`None`，表示暂不支持转换。在单测端可设置`run(unsupport=True, reason="")`来检测转换`Matcher`的正确性，其中`reason`表示不支持的原因(必填)。参考 [torch.median测试用例](https://github.com/PaddlePaddle/PaConvert/tree/master/tests/test_median.py)。对于不支持的转换且`Matcher`在转写层难以判断是否支持，例如不支持某种特定类型的输入，单测函数可以以`_`开头表示暂不运行该单测，同时需要注释不支持的原因。参考 [torch.addmm测试用例](https://github.com/PaddlePaddle/PaConvert/tree/master/tests/test_addmm.py)。

* **自定义检查**：如果需要自定义检查逻辑，可以继承 `APIBase` 类并重写`compare()`函数，实现自定义的检查逻辑，但需有充分理由，例如 `torch.Generator` 由于返回的不为Tensor，无法使用常规方法测试。 参考 [torch.Generator测试用例](https://github.com/PaddlePaddle/PaConvert/tree/master/tests/test_Generator.py)。

* **运行单测**：可以在主目录下执行`pytest tests`命令运行所有单测；也可以执行`pytest tests/xxx.py`运行`tests`目录下的某一单测；如果希望遇到`error`则停止单测，可以加上参数`-x`，即`pytest tests/test_add.py -x`，单测运行过程中会将转换后的`paddle`代码写入`test_project/paddle_temp.py`，方便排查错误。

**单测规范**：

* **精度与输入要求**：

单测的输入Tensor 必须规范，不能为全为0值等无效输入，默认会通过 `numpy.testing.assert_allclose(pytorch_result, paddle_result, rtol=1.0e-6, atol=0.0)` 来对比Pytorch、Paddle API的前向输出精度。

* **单测覆盖范围要求**：

单测本质为模仿用户Pytorch代码的用法，因此需要考虑该torch api所有可能的用法case，不可自行增加判断来刻意绕过测试，否则可能无法起到验证的作用。

单测覆盖范围要求为：涉及到多个API形参的，应包含各种用法（ `全部指定关键字、全部不指定关键字、改变关键字顺序、默认参数均不指定、参数取值以变量形式传入` 五种情况必须考虑），不能只考虑最简单常见的用法，首次增加1个单测文件至少列举5种不同的用法case（越多越好）。

以 `torch.Tensor.new_zeros` 为例，其至少包含12种以上的torch用法case，如下：

```python
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

总的来说，转换规则与单测的开发具有一定的挑战性，是一项需要非常细心的工作。
