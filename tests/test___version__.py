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
#

import textwrap

from apibase import APIBase


class VersionAPI(APIBase):
    """Equivalence check for ``torch.__version__`` / ``paddle.__version__``.

    ``torch.torch_version.TorchVersion`` is a ``str`` subclass whose
    comparison operators dispatch through ``packaging.version.Version``,
    falling back to plain string comparison on invalid versions. The Paddle
    equivalent must mirror that surface. Each framework parses ``Version``
    with its own copy of ``packaging`` (torch vendors it), so this test
    compares only against ``str`` / ``tuple`` inputs -- both implementations
    convert those through their own ``Version`` internally.
    """

    def compare(
        self,
        name,
        pytorch_result,
        paddle_result,
        check_value=True,
        check_shape=True,
        check_dtype=True,
        check_stop_gradient=True,
        rtol=1.0e-6,
        atol=0.0,
    ):
        assert isinstance(
            pytorch_result, str
        ), f"pytorch result should be a str (subclass), got {type(pytorch_result)}"
        assert isinstance(
            paddle_result, str
        ), f"paddle result should be a str (subclass), got {type(paddle_result)}"

        assert str(pytorch_result), "pytorch version string is empty"
        assert str(paddle_result), "paddle version string is empty"

        assert isinstance(pytorch_result.split("."), list)
        assert isinstance(paddle_result.split("."), list)
        assert pytorch_result.startswith(str(pytorch_result)[0])
        assert paddle_result.startswith(str(paddle_result)[0])

        assert (pytorch_result == str(pytorch_result)) is True
        assert (paddle_result == str(paddle_result)) is True

        assert (pytorch_result > "0.0.1") is True
        assert (paddle_result > "0.0.1") is True
        assert (pytorch_result > (0, 0, 1)) is True
        assert (paddle_result > (0, 0, 1)) is True
        assert (pytorch_result >= "0.0.1") is True
        assert (paddle_result >= "0.0.1") is True
        assert (pytorch_result < "999.0") is True
        assert (paddle_result < "999.0") is True

        assert (pytorch_result == "parrot") == (paddle_result == "parrot")
        assert (pytorch_result != "parrot") == (paddle_result != "parrot")

        # __eq__ is installed via setattr on both, which does not implicitly
        # reset __hash__ to None -- guard against that regression.
        hash(pytorch_result)
        hash(paddle_result)


obj = VersionAPI("torch.__version__")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.__version__
        """
    )
    obj.run(pytorch_code, ["result"])
