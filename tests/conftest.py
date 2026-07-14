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

"""Global-state isolation for the test suite.

Each test execs both a torch reference and the converted paddle code in the SAME
worker process (see ``tests/apibase.py``). Process-global switches such as
``torch.set_default_device('cuda')``, ``set_default_dtype`` or the grad-enabled
flag therefore LEAK into every later test in that process, causing failures that
only appear in the full suite (and vanish on a ``--lf`` rerun that no longer
schedules the polluting test first).

The autouse fixture below snapshots that global state before each test and
restores it afterwards, so a test that forgets to reset (e.g.
``torch.set_default_device('cuda:0')`` with no matching reset) can no longer
corrupt its neighbours.

The logic is lazy and best-effort: it only touches a framework that is already in
``sys.modules``, so it never forces an import of torch/paddle (which would change
import ordering) for tests that don't use them.
"""

import os
import sys

import pytest


def disable_paddle_compat():
    try:
        from paddle.compat.proxy import TORCH_PROXY_FINDER
    except (ImportError, ModuleNotFoundError):
        return

    if TORCH_PROXY_FINDER not in sys.meta_path:
        return

    try:
        import paddle

        paddle.disable_compat()
    except (AttributeError, ImportError, ModuleNotFoundError):
        if TORCH_PROXY_FINDER in sys.meta_path:
            sys.meta_path.remove(TORCH_PROXY_FINDER)


def _snapshot_environ():
    """Record the process environment so env-var writes can be reverted.

    Converted code maps some torch APIs to environment variables, e.g.
    ``torch.set_num_threads(n)`` -> ``os.environ['CPU_NUM'] = str(n)``. Left set,
    that leaks into ``torch.get_num_threads`` -> ``os.getenv('CPU_NUM', 1)`` in a
    later test, which then returns a ``str`` instead of the default ``int``.
    """
    return dict(os.environ)


def _restore_environ(snap):
    for key in list(os.environ.keys()):
        if key not in snap:
            try:
                del os.environ[key]
            except Exception:
                pass
    for key, value in snap.items():
        if os.environ.get(key) != value:
            try:
                os.environ[key] = value
            except Exception:
                pass


def _snapshot_torch():
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    context_holder = torch._GLOBAL_DEVICE_CONTEXT
    if getattr(context_holder, "device_context", False) is None:
        del context_holder.device_context
    snap = {}
    try:
        snap["dtype"] = torch.get_default_dtype()
    except Exception:
        pass
    try:
        snap["device"] = torch.get_default_device()
        snap["has_device_context"] = hasattr(
            torch._GLOBAL_DEVICE_CONTEXT, "device_context"
        )
    except Exception:
        pass
    try:
        snap["grad"] = torch.is_grad_enabled()
    except Exception:
        pass
    return snap


def _restore_torch(snap):
    torch = sys.modules.get("torch")
    if torch is None:
        return
    dtype = torch.float32
    device = "cpu"
    grad = True
    if snap:
        dtype = snap.get("dtype", dtype)
        device = snap.get("device", device)
        grad = snap.get("grad", grad)
    try:
        torch.set_default_dtype(dtype)
    except Exception:
        pass
    try:
        context_holder = torch._GLOBAL_DEVICE_CONTEXT
        current_context = getattr(context_holder, "device_context", None)
        if current_context is not None:
            current_context.__exit__(None, None, None)
        if hasattr(context_holder, "device_context"):
            del context_holder.device_context
        if snap and snap.get("has_device_context"):
            torch.set_default_device(device)
    except Exception:
        pass
    try:
        torch.set_grad_enabled(grad)
    except Exception:
        pass


def _snapshot_paddle():
    paddle = sys.modules.get("paddle")
    if paddle is None:
        return None
    snap = {}
    try:
        snap["dtype"] = paddle.get_default_dtype()
    except Exception:
        pass
    try:
        snap["device"] = paddle.device.get_device()
    except Exception:
        pass
    try:
        snap["grad"] = paddle.is_grad_enabled()
    except Exception:
        pass
    return snap


def _restore_paddle(snap):
    paddle = sys.modules.get("paddle")
    if paddle is None or not snap:
        return
    if "dtype" in snap:
        try:
            paddle.set_default_dtype(snap["dtype"])
        except Exception:
            pass
    if "device" in snap:
        try:
            paddle.device.set_device(snap["device"])
        except Exception:
            pass
    if "grad" in snap:
        try:
            paddle.set_grad_enabled(snap["grad"])
        except Exception:
            pass


_SENTINEL = object()


def _patch_targets():
    """Global paddle objects that converted code monkeypatches via ``setattr``.

    Converted paddle code frequently does e.g.
    ``setattr(paddle.Tensor, "add", _add)`` or
    ``setattr(paddle.nn.LogSoftmax, "forward", _log_softmax_forward)`` where the
    helper is defined in the exec namespace. Because the patch lands on the
    *global* class it leaks into every later test; and because ``apibase`` clears
    the exec namespace afterwards, the helper's ``__globals__`` is emptied, so a
    later call raises ``NameError: name 'paddle' is not defined``.

    We return ``paddle.Tensor`` plus every class exposed on ``paddle.nn`` so the
    fixture can snapshot their attributes and revert any test-induced change.
    """
    paddle = sys.modules.get("paddle")
    if paddle is None:
        return []
    targets = []
    tensor = getattr(paddle, "Tensor", None)
    if tensor is not None:
        targets.append(tensor)
    nn = getattr(paddle, "nn", None)
    if nn is not None:
        for name in dir(nn):
            try:
                obj = getattr(nn, name)
            except Exception:
                continue
            if isinstance(obj, type):
                targets.append(obj)
    return targets


def _snapshot_patches():
    """Record current attribute values of monkeypatchable paddle targets."""
    snap = []
    for target in _patch_targets():
        try:
            attrs = dict(vars(target))
        except TypeError:
            continue
        snap.append((target, attrs))
    return snap


def _restore_patches(snap):
    """Revert only the attributes a test added or replaced on paddle targets."""
    for target, attrs in snap:
        try:
            current = dict(vars(target))
        except TypeError:
            continue
        # Restore attributes whose object identity changed during the test.
        for name, original in attrs.items():
            if current.get(name, _SENTINEL) is not original:
                try:
                    setattr(target, name, original)
                except (AttributeError, TypeError):
                    pass
        # Delete attributes that the test added and that were not present before.
        for name in current:
            if name not in attrs:
                try:
                    delattr(target, name)
                except (AttributeError, TypeError):
                    pass


@pytest.fixture(autouse=True)
def _reset_global_state():
    """Snapshot and restore torch/paddle process-global state around each test.

    Covers the classes of cross-test pollution that only surface in the full
    suite (and vanish on ``--lf``): (1) torch/paddle default device, (2) default
    dtype and grad mode, (3) global monkeypatches converted code applies to
    ``paddle.Tensor`` / ``paddle.nn.*`` classes, and (4) environment variables
    such as ``CPU_NUM`` written by converted ``set_num_threads`` code.
    """
    disable_paddle_compat()
    torch_snap = _snapshot_torch()
    paddle_snap = _snapshot_paddle()
    patch_snap = _snapshot_patches()
    environ_snap = _snapshot_environ()
    try:
        yield
    finally:
        disable_paddle_compat()
        _restore_patches(patch_snap)
        _restore_torch(torch_snap)
        _restore_paddle(paddle_snap)
        _restore_environ(environ_snap)
