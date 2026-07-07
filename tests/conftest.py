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


def disable_paddle_compat():
    """Turn OFF Paddle's torch-compat proxy if it is currently active.

    Converted Paddle code injects ``paddle.enable_compat(level=2)``, which flips
    process-global state: it installs an ``import torch`` -> Paddle proxy and
    aliases ``paddle.*`` to the torch-aligned ``paddle.compat.*`` APIs. That state
    must be cleared so it cannot leak into (a) a later test, or (b) a torch
    *reference* run within the same test, whose ``import torch`` would otherwise be
    proxied to Paddle so the reference would no longer be real torch.

    Lazy and best-effort: a no-op when Paddle was never imported, so it does not
    force a Paddle import (and thus does not change torch/paddle import ordering)
    for tests that never touch Paddle.
    """
    if "paddle" not in sys.modules:
        return
    paddle = sys.modules["paddle"]
    try:
        from paddle.compat.proxy import TORCH_PROXY_FINDER

        while TORCH_PROXY_FINDER in sys.meta_path:
            paddle.disable_compat()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _reset_paddle_compat_mode():
    """Disable torch-compat after every test so it cannot leak into a later one."""
    yield
    disable_paddle_compat()
