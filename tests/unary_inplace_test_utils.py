# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
#

import re
import textwrap


def register_standard_unary_inplace_tests(
    namespace,
    obj,
    method_name,
    case_data_2d,
    case_data_3d,
):
    existing_indices = []
    for name in namespace:
        match = re.fullmatch(r"test_case_(\d+)", name)
        if match:
            existing_indices.append(int(match.group(1)))

    next_index = max(existing_indices, default=0) + 1

    cases = [
        (
            f"""
            import torch
            a = torch.tensor({case_data_2d}, dtype=torch.float64)
            result = a.{method_name}()
            same_object = result is a
            """,
            ["result", "a", "same_object"],
        ),
        (
            f"""
            import torch
            a = torch.tensor({case_data_3d}, dtype=torch.float32)
            result = a.{method_name}()
            same_object = result is a
            """,
            ["result", "a", "same_object"],
        ),
        (
            f"""
            import torch
            a = torch.tensor({case_data_2d}, dtype=torch.float32)
            alias = a
            result = alias.{method_name}()
            same_object = result is alias
            """,
            ["result", "a", "alias", "same_object"],
        ),
        (
            f"""
            import torch
            base = torch.tensor({case_data_2d}, dtype=torch.float32)
            a = base.t()
            result = a.{method_name}()
            same_object = result is a
            """,
            ["result", "a", "base", "same_object"],
        ),
        (
            f"""
            import torch
            a = torch.empty([0], dtype=torch.float32)
            result = a.{method_name}()
            same_object = result is a
            """,
            ["result", "a", "same_object"],
        ),
    ]

    for pytorch_code, compared_tensor_names in cases:

        def test_case(code=pytorch_code, names=compared_tensor_names):
            obj.run(textwrap.dedent(code), names)

        test_case.__name__ = f"test_case_{next_index}"
        namespace[test_case.__name__] = test_case
        next_index += 1
