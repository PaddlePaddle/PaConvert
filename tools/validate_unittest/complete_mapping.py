# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import importlib
import inspect
import json
from collections import OrderedDict


def isclassname(name, module_parts):
    if name[0].isupper():
        return True
    elif (
        name == "profile" and len(module_parts) >= 1 and module_parts[-1] == "profiler"
    ):
        return True
    return False


def find_function_by_string(function_string):
    try:
        parts = function_string.split(".")
        function_name = parts.pop()
        classname = None
        if len(parts) >= 1 and isclassname(parts[-1], parts[:-1]):
            classname = parts.pop()

        if len(parts) > 0:
            module_name = ".".join(parts)
            module = importlib.import_module(module_name)
        else:
            module = globals()

        if classname is not None:
            module = getattr(module, classname)

        if not hasattr(module, function_name):
            raise AttributeError(f"{function_name} is not in {module_name}")

        return getattr(module, function_name)
    except AttributeError as e:
        print(f"skip finding function {function_string}: {e}")
        return None
    except Exception as e:
        print(f"Error loading function {function_string}: {e}")
        return None


def get_non_default_positional_args(func):
    signature = inspect.signature(func)
    non_default_args = []

    for param_name, param in signature.parameters.items():
        if (
            param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            or param.kind == inspect.Parameter.POSITIONAL_ONLY
        ):
            if param.default == inspect.Parameter.empty:
                non_default_args.append(param_name)

    return non_default_args


if __name__ == "__main__":
    with open("paconvert/api_mapping.json", "r") as f:
        api_mapping = json.load(f, object_pairs_hook=OrderedDict)

    count = 0

    for k in api_mapping:
        v = api_mapping[k]
        if "min_input_args" not in v:
            founded_function = find_function_by_string(k)
            if founded_function is None:
                continue
            try:
                non_default_args = get_non_default_positional_args(founded_function)
                if len(non_default_args) > 0 and non_default_args[0] == "self":
                    non_default_args.pop(0)
                print(f"{k}: {non_default_args}")
                count += 1
                api_mapping[k]["min_input_args"] = len(non_default_args)
            except (ValueError, TypeError) as e:
                print(f"Error loading function {k}: {e}")

    if count > 0:
        with open("paconvert/api_mapping.json", "w") as f:
            json.dump(api_mapping, f, indent=2)
        print(f"update {count} `min_input_args` field")
