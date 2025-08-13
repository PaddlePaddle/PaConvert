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
import threading

import isort


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


class UtilsFileHelper(object):
    _instance = None
    _lock = threading.Lock()

    START_CONTENT = (
        "\n############################## 相关utils函数，如下 ##############################"
    )
    INIT_CONTENT = (
        "############################ PaConvert 自动生成的代码 ###########################"
    )

    END_CONTENT = (
        "############################## 相关utils函数，如上 ##############################\n\n"
    )

    def __init__(self, fileName=None, is_dir_mode=False, logger=None):
        if not hasattr(self, "initialized"):
            super().__init__()
            self.fileName = fileName
            self.is_dir_mode = is_dir_mode
            self.code_map = {}
            self.initialized = True
            self.logger = logger

    def __new__(cls, fileName=None, is_dir_mode=False, logger=None):
        if cls._instance is None:
            with cls._lock:
                cls._instance = super().__new__(cls)
        return cls._instance

    def _get_code_hash(self, code: str) -> int:
        base_hash = hash(code)
        while base_hash in self.code_map and self.code_map[base_hash] != code:
            base_hash = hash(f"{code}_{len(self.code_map)}")
        return base_hash

    def add_code(self, code: str) -> int:
        """
        Add the code to the code map and return the code hash
        """
        if not self.fileName:
            return None

        code = code.rstrip("\n")
        code_hash = self._get_code_hash(code)
        if code_hash not in self.code_map:
            self.code_map[code_hash] = code
        return code_hash

    def write_code(self):
        """
        Write all the code in the code map to destination file
        """
        if not self.fileName:
            return

        if not self.code_map:
            return

        code_lines = []
        if self.is_dir_mode:
            code_lines += ["\nimport paddle\n", self.START_CONTENT, self.INIT_CONTENT]
        else:
            code_lines += [self.START_CONTENT]
        code_lines += self.code_map.values()
        code_lines.append(self.END_CONTENT)
        insert_code = "\n".join(code_lines)

        # insert the new code into the existing file
        if not self.is_dir_mode:
            with open(self.fileName, "r") as f:
                existing_content = f.read()

            # find a position to insert the new code
            lines = existing_content.splitlines()
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("import ") or line.startswith("from "):
                    insert_idx = i + 1

            # insert the new code after all imports
            new_content = "".join(
                [
                    "\n".join(lines[:insert_idx]),
                    insert_code,
                    "\n".join(lines[insert_idx:]),
                ]
            )
        else:
            os.makedirs(os.path.dirname(self.fileName), exist_ok=True)
            new_content = insert_code

        # remove redundant import
        try:
            new_content = isort.code(new_content)
        except Exception as e:
            log_info(
                self.logger,
                "Skip isort format due to error: {}".format(str(e)),
            )

        # write to file
        with open(self.fileName, "w", encoding="UTF-8") as f:
            f.write(new_content)
        self.code_map.clear()


def log_error(logger, msg, file=None, line=None):
    if file:
        if line:
            msg = "[{}:{}] {}".format(file, line, msg)
        else:
            msg = "[{}] {}".format(file, msg)
    else:
        msg = "{}".format(msg)
    logger.error(msg)


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
