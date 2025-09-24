# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import textwrap

from apibase import APIBase

obj = APIBase("torch.nn.attention.sdpa_kernel")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        
        modified_backend_state = [torch.nn.attention.SDPBackend.MATH]
        
        np.random.seed(100)
        x_data = np.random.randn(2, 2, 2, 2)
        x = torch.tensor(x_data, dtype=torch.float32)
        
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            result = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(torch.float32)
            current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
            assert current_backends == modified_backend_state
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1e-3)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        
        original_backend_state = set(torch.nn.attention._cur_sdpa_kernel_backends())
        modified_backend_state = [torch.nn.attention.SDPBackend.MATH]
        
        np.random.seed(100)
        x_data = np.random.randn(2, 2, 2, 2)
        x = torch.tensor(x_data, dtype=torch.float32)
        
        # Check original state
        current_backends = set(torch.nn.attention._cur_sdpa_kernel_backends())
        assert current_backends == original_backend_state, f"Expected {original_backend_state}, got {current_backends}"
        
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            output1 = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(torch.float32)
            current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
            assert current_backends == modified_backend_state, f"Expected {modified_backend_state}, got {current_backends}"
            
            output2 = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(torch.float32)
            current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
            assert current_backends == modified_backend_state, f"Expected {modified_backend_state}, got {current_backends}"
            
            output3 = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(torch.float32)
            current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
            assert current_backends == modified_backend_state, f"Expected {modified_backend_state}, got {current_backends}"
        
        # Check back to original state
        current_backends = set(torch.nn.attention._cur_sdpa_kernel_backends())
        assert current_backends == original_backend_state, f"Expected {original_backend_state}, got {current_backends}"
        
        result = output1 + output2 + output3
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1e-3)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        
        modified_backend_state = {
            torch.nn.attention.SDPBackend.MATH,
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        }
        
        np.random.seed(100)
        x_data = np.random.randn(2, 2)
        x = torch.tensor(x_data, dtype=torch.float32)
        
        with torch.nn.attention.sdpa_kernel([
            torch.nn.attention.SDPBackend.MATH,
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        ]):
            # FLASH_ATTENTION may not be supported, but we're not actually doing any sdpa
            x = x + 1
            current_backends = set(torch.nn.attention._cur_sdpa_kernel_backends())
            assert current_backends == modified_backend_state, f"Expected {modified_backend_state}, got {current_backends}"
            x = x + 1
        
        result = x
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        
        backends = [torch.nn.attention.SDPBackend.MATH]
        
        np.random.seed(100)
        x_data = np.random.randn(2, 2)
        x = torch.tensor(x_data, dtype=torch.float32)
        
        with torch.nn.attention.sdpa_kernel(backends=backends, set_priority=True):
            x = x + 1
            current_backends = set(torch.nn.attention._cur_sdpa_kernel_backends())
            expected_backends = set(backends)
            assert current_backends == expected_backends, f"Expected {expected_backends}, got {current_backends}"
            x = x + 1
        
        result = x
        """
    )
    obj.run(pytorch_code, ["result"])

def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        
        backends = [torch.nn.attention.SDPBackend.MATH]
        
        np.random.seed(100)
        x_data = np.random.randn(2, 2)
        x = torch.tensor(x_data, dtype=torch.float32)
        
        with torch.nn.attention.sdpa_kernel(backends=backends, set_priority=True):
            x = x + 1
            current_backends = set(torch.nn.attention._cur_sdpa_kernel_backends())
            expected_backends = set(backends)
            assert current_backends == expected_backends, f"Expected {expected_backends}, got {current_backends}"
            x = x + 1
        
        result = x
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        
        backends = [torch.nn.attention.SDPBackend.MATH]
        
        np.random.seed(100)
        x_data = np.random.randn(2, 2)
        x = torch.tensor(x_data, dtype=torch.float32)
        
        with torch.nn.attention.sdpa_kernel(backends=backends, set_priority=False):
            x = x + 1
            current_backends = set(torch.nn.attention._cur_sdpa_kernel_backends())
            expected_backends = set(backends)
            assert current_backends == expected_backends, f"Expected {expected_backends}, got {current_backends}"
            x = x + 1
        
        result = x
        """
    )
    obj.run(pytorch_code, ["result"])
