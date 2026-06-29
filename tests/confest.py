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

import sys

import pytest


@pytest.fixture(autouse=True)
def _reset_paddle_compat_mode():
    """Converted Paddle code node injects ``paddle.enable_compat()``, which flips
    global state: it installs an ``import torch`` -> Paddle proxy adn aliases
    ``paddle.*`` to the torch-aligned ``paddle.compat.*`` APIs. Disable it after
    every test so it cannot leak into a later test (e.g., corrupt another test's)
    real-torch reference run, or break tests that assume compat is off.

    Lazy and best-effort: a no-op when Paddle was never imported, so it does not
    force a Paddle import (and thus does not change torch/paddle import ordering)
    for tests that never touch Paddle
    """
    yield
    if "paddle" not in sys.modules:
        return
    paddle = sys.modules["paddle"]
    try:
        from paddle.compat.proxy import TORCH_PROXY_FINDER

        while TORCH_PROXY_FINDER in sys.meta_path:
            paddle.disable_compat()
    except Exception:
        pass
