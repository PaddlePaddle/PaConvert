# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import textwrap

from apibase import APIBase

obj = APIBase("transformers.StoppingCriteriaList")

# def test_case_1():
#     pytorch_code = textwrap.dedent(
#         """
#         import torch
#         from transformers import StoppingCriteriaList

#         class MockStoppingCriteria:
#             def __init__(self, target_length):
#                 self.target_length = target_length

#             def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
#                 is_done = input_ids.shape[-1] >= self.target_length
#                 return torch.full((input_ids.shape[0],), is_done, device=input_ids.device, dtype=torch.bool)

#         criteria_list = StoppingCriteriaList([MockStoppingCriteria(target_length=5)])
#         input_ids = torch.ones(2, 4, dtype=torch.long)
#         scores = torch.rand(2, 100)
#         result_1 = criteria_list(input_ids, scores)
#         input_ids_2 = torch.ones(2, 5, dtype=torch.long)
#         result_2 = criteria_list(input_ids_2, scores)

#         is_stopped_1 = result_1.tolist()
#         is_stopped_2 = result_2.tolist()
#         """
#     )
#     obj.run(pytorch_code, ["is_stopped_1", "is_stopped_2"],
#         unsupport=True,
#         reason="paddleformers bug")

# def test_case_2():
#     pytorch_code = textwrap.dedent(
#         """
#         import torch
#         from transformers import StoppingCriteriaList

#         class MockStoppingCriteria:
#             def __init__(self, target_length):
#                 self.target_length = target_length

#             def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
#                 is_done = input_ids.shape[-1] >= self.target_length
#                 return torch.full((input_ids.shape[0],), is_done, device=input_ids.device, dtype=torch.bool)

#         criteria_A = MockStoppingCriteria(target_length=6)
#         criteria_B = MockStoppingCriteria(target_length=4)

#         criteria_list = StoppingCriteriaList([criteria_A, criteria_B])
#         input_ids = torch.ones(2, 5, dtype=torch.long)
#         scores = torch.rand(2, 100)
#         result = criteria_list(input_ids, scores)
#         is_stopped = result.tolist()
#         """
#     )
#     obj.run(pytorch_code, ["is_stopped"],
#         unsupport=True,
#         reason="paddleformers bug")


def test_case_3_max_length_property():
    pytorch_code = textwrap.dedent(
        """
        from transformers import StoppingCriteriaList

        class MockStoppingCriteria:
            def __init__(self, target_length):
                self.target_length = target_length

            def __call__(self, input_ids, scores, **kwargs):
                return torch.full((input_ids.shape[0],), False)

        class MaxLengthCriteria:
            def __init__(self, max_length):
                self.max_length = max_length

            def __call__(self, input_ids, scores, **kwargs):
                return torch.full((input_ids.shape[0],), False)

        list_A = StoppingCriteriaList([MockStoppingCriteria(10), MaxLengthCriteria(max_length=50)])
        max_length_A = list_A.max_length

        list_B = StoppingCriteriaList([MockStoppingCriteria(10)])
        max_length_B = list_B.max_length

        list_C = StoppingCriteriaList([])
        max_length_C = list_C.max_length
        """
    )
    obj.run(pytorch_code, ["max_length_A", "max_length_B", "max_length_C"])
