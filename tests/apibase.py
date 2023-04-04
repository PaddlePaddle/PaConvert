import os
import collections

import ast
import astor
import logging
from paconvert.converter import Converter

from paconvert.transformer.basic_transformer import BasicTransformer
from paconvert.transformer.import_transformer import ImportTransformer

class APIBase(object):
    def __init__(self, pytorch_api) -> None:
        """
        args:
            pytorch_api: The corresponding pytorch api
        """
        self.pytorch_api = pytorch_api
        self.logger = logging.getLogger(name='API Test')
        self.imports_map = collections.defaultdict(dict)
        self.unsupport_map = collections.defaultdict(int)
        pass

    def run(self, pytorch_code, args, file_name) -> None:
        """
        args:
            pytorch_code: pytorch code to execute
            args: the list of variant to be checked
            file_name: current test file name
        """
        loc = locals()
        exec(pytorch_code)
        pytorch_result = [loc[arg] for arg in args]

        paddle_code = self.convert(pytorch_code, file_name)
        exec(paddle_code)
        paddle_result = [loc[arg] for arg in args]
        for i in range(len(args)):
            assert self.check(pytorch_result[i], paddle_result[i]), '[{}]: convert failed'.format(self.pytorch_api)

    def check(self, pytorch_result, paddle_result):
        """
        compare tensors' data, requires_grad, dtype
        args:
            pytorch_result: pytorch Tensor
            paddle_result: paddle Tensor
        """
        torch_numpy, paddle_numpy = pytorch_result.numpy(), paddle_result.numpy()
        if not (torch_numpy == paddle_numpy).all():
            return False
        if pytorch_result.requires_grad == paddle_result.stop_gradient:
            return False
        if str(pytorch_result.dtype)[6:] != str(paddle_result.dtype)[7:]:
            return False
        return True

    def convert(self, pytorch_code, file_name):
        """
        convert pytorch code to paddle code.
        args:
            pytorch_code: pytorch code to be converted.
            file_name: name of file to be converted.
        return:
            paddle code.
        """
        # the converted paddle code will be temporarily written to this file.
        new_file_name = os.getcwd() + '/paddle_project/pytorch_temp.py'

        root = ast.parse(pytorch_code)
        import_trans = ImportTransformer(root, file_name, self.imports_map, self.logger)
        import_trans.transform()

        if import_trans.import_paddle:
            api_trans = BasicTransformer(root, file_name, self.imports_map, self.logger, self.unsupport_map)
            api_trans.transform()

        code = astor.to_source(root)
        code = Converter.mark_unsport(self, code)

        with open(new_file_name, 'w', encoding='UTF-8') as new_file:
            new_file.write(code)
        return code