# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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

import collections
import os
import re
import textwrap


class UniqueNameGenerator:
    def __init__(self):
        self.ids = collections.defaultdict(int)

    def __call__(self, key):
        counter = self.ids[key]
        self.ids[key] += 1
        return "_".join([key, str(counter)])


Generator = UniqueNameGenerator()


def get_unique_name(key):
    return Generator(key)


class AuxFileHelper(object):
    _instance = None
    INIT_CONTENT = """
    ############################## 相关utils函数，如下 ##############################
    ####################### PaConvert 自动生成的代码，请勿手动修改! ##################
    import paddle
    """
    END_CONTENT = """
    ############################## 相关utils函数，如上 ##############################
    """

    def __init__(self, fileName=None, is_dir_mode=False):
        if not hasattr(self, "initialized"):
            super().__init__()
            self.fileName = fileName
            self.is_dir_mode = is_dir_mode
            self.ids = collections.defaultdict(int)
            self.code_map = {}
            self.initialized = True

    def __new__(cls, fileName=None, is_dir_mode=False):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_code_hash(self, code: str) -> int:
        base_hash = hash(code)
        while base_hash in self.code_map and self.code_map[base_hash] != code:
            base_hash = hash(f"{code}_{len(self.ids)}")
        return base_hash

    def add_code(self, code: str, torch_api: str) -> int:
        """
        Add the code to the code map and return the code hash
        """
        if not self.fileName:
            return None

        code_hash = self._get_code_hash(code)
        if self.ids[code_hash] == 0:
            self.code_map[code_hash] = code
        self.ids[code_hash] += 1
        return code_hash

    def write_code(self):
        """
        Write all the code in the code map to destination file
        """
        if not self.fileName:
            return

        if not self.code_map:
            return

        all_code = textwrap.dedent(self.INIT_CONTENT) + "\n"
        all_code += "".join(self.code_map.values())
        all_code += textwrap.dedent(self.END_CONTENT) + "\n"

        # insert the new code into the existing file
        if os.path.exists(self.fileName):
            with open(self.fileName, "r") as f:
                existing_content = f.read()

            # only rewrite the content between the INIT_CONTENT and END_CONTENT
            start_marker = "############################## 相关utils函数，如下 ##############################"
            end_marker = "############################## 相关utils函数，如上 ##############################"
            start_index = existing_content.find(start_marker)
            end_index = existing_content.find(end_marker)
            if start_index != -1 and end_index != -1:
                end_index += len(end_marker)
                existing_content = (
                    existing_content[:start_index]
                    + existing_content[end_index:].lstrip()
                )

            # remove the existing paddle import
            if re.search(r"^import\s+paddle\s*$", existing_content, re.MULTILINE):
                all_code = all_code.replace("import paddle\n", "")

            # find the last import statement to insert the new code after it
            lines = existing_content.splitlines()
            insert_line = 0
            last_import_line = -1
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("import ") or line.startswith("from "):
                    last_import_line = i
                    continue

            # insert the new code after the last import statement
            insert_line = last_import_line + 1 if last_import_line != -1 else 0
            new_content = "\n".join(lines[:insert_line]) + "\n\n"
            new_content += all_code
            new_content += "\n".join(lines[insert_line:])

            with open(self.fileName, "w") as f:
                f.write(new_content)
        else:
            os.makedirs(os.path.dirname(self.fileName), exist_ok=True)
            with open(self.fileName, "w") as f:
                f.write(all_code)


def log_warning(logger, msg, file=None, line=None):
    if file:
        if line:
            msg = "[{}:{}] {}".format(file, line, msg)
        else:
            msg = "[{}] {}".format(file, msg)
    else:
        msg = "{}".format(msg)
    logger.warning(msg)


def log_info(logger, msg, file=None, line=None):
    if file:
        if line:
            msg = "[{}:{}] {}".format(file, line, msg)
        else:
            msg = "[{}] {}".format(file, msg)
    else:
        msg = "{}".format(msg)
    logger.info(msg)


def log_debug(logger, msg, file=None, line=None):
    if file:
        if line:
            msg = "[{}:{}] {}".format(file, line, msg)
        else:
            msg = "[{}] {}".format(file, msg)
    else:
        msg = "{}".format(msg)
    logger.debug(msg)


def process_reduce_and_size_average(kwargs):
    if "size_average" in kwargs:
        size_average = kwargs.pop("size_average")
        if "True" in size_average:
            size_average = True
        elif "False" in size_average:
            size_average = False
        else:
            size_average = None
    else:
        size_average = None

    if "reduce" in kwargs:
        reduce = kwargs.pop("reduce")
        if "True" in reduce:
            reduce = True
        elif "False" in reduce:
            reduce = False
        else:
            reduce = None
    else:
        reduce = None

    if size_average is not None or reduce is not None:
        if size_average is None:
            size_average = True
        if reduce is None:
            reduce = True

        if size_average and reduce:
            reduction = '"""mean"""'
        elif reduce:
            reduction = '"""sum"""'
        else:
            reduction = '"""none"""'

        kwargs["reduction"] = reduction
