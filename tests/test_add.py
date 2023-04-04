
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase


class TestTorchAdd(APIBase):
    def __init__(self, pytorch_api) -> None:
        super().__init__(pytorch_api)
        pass

    def check(self, res_pytorch, res_paddle):
        """
        execute customize the logic of check or the superclass's
        """
        return True

obj = APIBase('torch.add')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.add(torch.tensor([1, 2, 3]), torch.tensor([1, 4, 6]))
        '''
    )
    obj.run(pytorch_code, ['result'])