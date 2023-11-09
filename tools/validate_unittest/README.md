## Validate Unittest

单测 api 调用多样性检查，用于检查是否满足以下情况：

- 全部不指定关键字 `all args`
- 全部指定关键字 `all kwargs`
- 改变关键字顺序 `kwargs out of order`
- 全部不指定默认值 `all default`

### 用法

首次使用时，需要先构建测试数据，在项目根目录下执行：

```bash
python tools/validate_unittest/validate_unittest.py -r tests
```

*注意：即使单测出错，也可以正常收集数据。等待全部执行完即可，会生成 `tools/validate_unittest/validation.json` 数据文件*

此时默认会生成 `tools/validate_unittest/validation_report.md` 作为多样性检测报告，其中仅包含多样性不符合的 api。

当修改后需要**重新检测**时，用法和 `pytest` 基本一致，支持三种更新方式：

1. 全局重新生成

    ```bash
    python tools/validate_unittest/validate_unittest.py -r tests
    ```

2. 重新生成单个单测的数据

    ```bash
    python tools/validate_unittest/validate_unittest.py -r tests/test_Tensor_amax.py
    ```

3. 重新生成若干个单测的数据

    ```bash
    # 通配符
    python tools/validate_unittest/validate_unittest.py -r tests/test_Tensor_div*
    # 列表
    python tools/validate_unittest/validate_unittest.py -r tests/test_Tensor_divide.py tests/test_Tensor_div.py
    ```
