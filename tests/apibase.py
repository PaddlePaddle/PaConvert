import os

import numpy as np

from paconvert.converter import Converter


class APIBase(object):
    def __init__(self, pytorch_api) -> None:
        """
        args:
            pytorch_api: The corresponding pytorch api
        """
        self.pytorch_api = pytorch_api

    def run(self, pytorch_code, compared_tensor_names=None,  expect_paddle_code=None) -> None:
        """
        args:
            pytorch_code: pytorch code to execute
            compared_tensor_names: the list of variant name to be compared
            expect_paddle_code: the string of expect paddle code
        """
        if compared_tensor_names:
            loc = locals()
            exec(pytorch_code)
            pytorch_result = [loc[name] for name in compared_tensor_names]

            paddle_code = self.convert(pytorch_code)
            exec(paddle_code)
            paddle_result = [loc[name] for name in compared_tensor_names]
            for i in range(len(compared_tensor_names)):
                assert self.check(pytorch_result[i], paddle_result[i]), '[{}]: convert failed'.format(self.pytorch_api)

        if expect_paddle_code:
            convert_paddle_code = self.convert(pytorch_code)
            assert convert_paddle_code == expect_paddle_code, '[{}]: convert failed'.format(self.pytorch_api)

    def check(self, pytorch_result, paddle_result):
        """
        compare tensors' data, shape, requires_grad, dtype
        args:
            pytorch_result: pytorch Tensor
            paddle_result: paddle Tensor
        """
        torch_numpy, paddle_numpy = pytorch_result.numpy(), paddle_result.numpy()

        if not np.allclose(paddle_numpy, torch_numpy):
            return False
        if pytorch_result.requires_grad == paddle_result.stop_gradient:
            return False
        if str(pytorch_result.dtype)[6:] != str(paddle_result.dtype)[7:]:
            return False
        return True

    def convert(self, pytorch_code):
        """
        convert pytorch code to paddle code.
        args:
            pytorch_code: pytorch code to be converted.
        return:
            paddle code.
        """
        if not os.path.exists(os.getcwd() + '/paddle_project'):
            os.makedirs(os.getcwd() + '/paddle_project')

        pytorch_code_path = os.getcwd() + '/paddle_project/pytorch_temp.py'
        paddle_code_path = os.getcwd() + '/paddle_project/paddle_temp.py'
        with open(pytorch_code_path, 'w', encoding='UTF-8') as f:
            f.write(pytorch_code)

        coverter = Converter(log_dir='disable', show_unsupport=True)
        coverter.run(pytorch_code_path, paddle_code_path)

        with open(paddle_code_path, 'r', encoding='UTF-8') as f:
            code = f.read()
        return code