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
"""
   isort:skip_file
"""

import ast
import collections
import logging
import os
import re
import shutil
import black
import isort
import astor

from paconvert.transformer.basic_transformer import BasicTransformer
from paconvert.transformer.import_transformer import ImportTransformer
from paconvert.transformer.tensor_requires_grad_transformer import (
    TensorRequiresGradTransformer,
)
from paconvert.transformer.custom_op_transformer import (
    PreCustomOpTransformer,
    CustomOpTransformer,
)
from paconvert.utils import (
    UtilsFileHelper,
    get_unique_name,
    log_info,
    log_warning,
)


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith("."):
            yield f


class Converter:
    def __init__(
        self,
        log_dir=None,
        log_level="INFO",
        log_markdown=False,
        show_all_api=False,
        show_unsupport_api=False,
        no_format=False,
        calculate_speed=False,
        only_complete=False,
    ):
        self.imports_map = collections.defaultdict(dict)
        self.torch_api_count = 0
        self.success_api_count = 0
        self.logger = logging.getLogger(name=get_unique_name("Converter"))
        if log_dir is None:
            self.logger.addHandler(logging.StreamHandler())
        elif log_dir == "disable":
            logging.disable(1)
        else:
            self.logger.addHandler(logging.FileHandler(log_dir, mode="w"))
        self.logger.setLevel(log_level)
        self.log_markdown = log_markdown
        self.show_all_api = show_all_api
        self.all_api_map = collections.defaultdict(dict)
        self.show_unsupport_api = show_unsupport_api
        self.unsupport_api_map = collections.defaultdict(int)
        self.convert_rate = 0.0
        self.no_format = no_format
        self.calculate_speed = calculate_speed
        self.only_complete = only_complete
        self.line_count = 0

        log_info(self.logger, "===========================================")
        log_info(self.logger, "PyTorch to Paddle Convert Start ------>:")
        log_info(self.logger, "===========================================")

    def run(self, in_dir, out_dir=None, exclude=None):
        if self.calculate_speed:
            import time

            start_time = time.time()

        in_dir = os.path.abspath(in_dir)
        if out_dir is None:
            out_dir = os.getcwd() + "/paddle_project"
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        else:
            out_dir = os.path.abspath(out_dir)

        assert out_dir != in_dir, "--out_dir must be different from --in_dir"

        if exclude:
            exclude = exclude.split(",")
        else:
            exclude = []
        exclude.append("__pycache__")

        if os.path.isfile(in_dir):
            out_file = (
                os.path.join(out_dir, os.path.basename(in_dir))
                if os.path.isdir(out_dir)
                else out_dir
            )
            utils_file_helper = UtilsFileHelper(
                out_file, is_dir_mode=False, logger=self.logger
            )
        elif os.path.isdir(in_dir):
            utils_file_helper = UtilsFileHelper(
                out_dir + "/paddle_utils.py", is_dir_mode=True, logger=self.logger
            )

        self.transfer_dir(in_dir, out_dir, exclude)
        utils_file_helper.write_code()

        if self.show_unsupport_api:
            log_info(self.logger, "\n===========================================")
            log_info(self.logger, "Not Support API List:")
            log_info(self.logger, "===========================================")
            if len(self.unsupport_api_map) == 0:
                log_info(
                    self.logger,
                    "Congratulations! All APIs have been successfully converted!",
                )
            else:
                log_info(
                    self.logger,
                    "These Pytorch APIs are not supported to convert to Paddle now, which will be supported in future!\n",
                )
                unsupport_api_list = sorted(
                    self.unsupport_api_map.items(), key=lambda x: x[1], reverse=True
                )
                for k, v in unsupport_api_list:
                    log_info(self.logger, "{:<80}{:<8}".format(k, v))

                import pandas

                df = pandas.DataFrame(
                    unsupport_api_list, columns=["PyTorch API", "Count"]
                )
                df.to_excel("unsupport_api_map.xlsx", index=False)

        if self.show_all_api:
            log_info(self.logger, "\n===========================================")
            log_info(self.logger, "ALL API List:")
            log_info(self.logger, "===========================================")
            if len(self.all_api_map) == 0:
                log_info(self.logger, "There is no API need to convert.")
            else:
                log_info(
                    self.logger,
                    "All APIs to be converted are as follows.\n",
                )
                all_api_list = sorted(
                    self.all_api_map.items(), key=lambda x: x[1]["count"], reverse=True
                )
                for k, v in all_api_list:
                    log_info(
                        self.logger,
                        "{:<80}{:<80}{:<8}".format(k, str(v["paddle_api"]), v["count"]),
                    )

                import pandas

                data = [(k, v["paddle_api"], v["count"]) for k, v in all_api_list]
                df = pandas.DataFrame(
                    data, columns=["PyTorch API", "Paddle API", "Count"]
                )
                df.to_excel("all_api_map.xlsx", index=False)

        faild_api_count = self.torch_api_count - self.success_api_count
        if not self.log_markdown:
            log_warning(self.logger, "\n===========================================")
            log_warning(self.logger, "Convert Summary")
            log_warning(self.logger, "===========================================")
            log_warning(
                self.logger,
                "There are {} Pytorch APIs in this Project:".format(
                    self.torch_api_count
                ),
            )
            log_warning(
                self.logger,
                " {}  Pytorch APIs have been converted to Paddle successfully!".format(
                    self.success_api_count
                ),
            )
            log_warning(
                self.logger,
                " {}  Pytorch APIs are not supported to convert to Paddle currently!".format(
                    faild_api_count
                ),
            )
            if self.torch_api_count > 0:
                self.convert_rate = self.success_api_count / self.torch_api_count
            log_warning(
                self.logger, " Convert Rate is: {:.2%}".format(self.convert_rate)
            )
            if faild_api_count > 0:
                log_warning(
                    self.logger,
                    "\nFor these {} Pytorch APIs that currently do not support to convert, which have been marked by >>> before the line, \nplease refer to "
                    "[https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html]"
                    " \nand convert it by yourself manually. In addition, these APIs will be supported in future.".format(
                        faild_api_count
                    ),
                )
            log_warning(
                self.logger,
                "\nThank you to use Paddle Code Convert Tool. You can make any suggestions \nto us by submitting issues to [https://github.com/PaddlePaddle/PaConvert].\n",
            )
        else:
            log_warning(self.logger, "# 转换总结")
            if self.torch_api_count > 0:
                self.convert_rate = self.success_api_count / self.torch_api_count
            log_warning(
                self.logger,
                "总计有 {} 行Pytorch代码需要被转换，整体转换率是 {:.2%}，数据如下：".format(
                    self.torch_api_count, self.convert_rate
                ),
            )
            log_warning(
                self.logger,
                "* {} 行Pytorch 代码被成功转换为飞桨".format(self.success_api_count),
            )
            log_warning(
                self.logger,
                "* {} 行Pytorch 代码暂时不支持自动转换为飞桨，因为飞桨不支持该功能，请手动转换".format(faild_api_count),
            )
            if faild_api_count > 0:
                log_warning(
                    self.logger,
                    "\n对于这{}行未转换的Pytorch代码，已经在行前通过 >>>>>> 标记，请参考[Pytorch-Paddle API映射表]"
                    "(https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html)来手工转换。"
                    "未来飞桨完善此部分功能后，这些API将被支持自动转换。\n".format(faild_api_count),
                )

        if self.calculate_speed:
            end_time = time.time()
            cost_time = end_time - start_time
            speed = self.line_count / cost_time
            log_info(
                self.logger,
                "\nThe total lines of code is {}, The total conversion time is {:.4f} s, Convert Speed is {:.4f} lines per second.\n".format(
                    self.line_count, cost_time, speed
                ),
            )

        return self.success_api_count, faild_api_count

    def transfer_dir(self, in_dir, out_dir, exclude):
        if os.path.isfile(in_dir):
            old_path = in_dir
            if exclude:
                for pattern in exclude:
                    if re.search(pattern, old_path):
                        return

            if os.path.isdir(out_dir):
                new_path = os.path.join(out_dir, os.path.basename(old_path))
            else:
                new_path = out_dir
            if not os.path.exists(os.path.dirname(new_path)):
                os.makedirs(os.path.dirname(new_path))
            if not os.path.isdir(os.path.dirname(new_path)):
                os.remove(os.path.dirname(new_path))
                os.makedirs(os.path.dirname(new_path))
            self.transfer_file(old_path, new_path)
        elif os.path.isdir(in_dir):
            in_dir_item = listdir_nohidden(in_dir)
            for item in in_dir_item:
                old_path = os.path.join(in_dir, item)
                new_path = os.path.join(out_dir, item)

                is_exclude = False
                if exclude:
                    for pattern in exclude:
                        if re.search(pattern, old_path):
                            is_exclude = True
                if is_exclude:
                    continue

                if os.path.isdir(old_path):
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)

                self.transfer_dir(old_path, new_path, exclude)
        elif os.path.islink(in_dir):
            # may need to create link
            pass
        else:
            raise ValueError(" the input 'in_dir' must be a exist file or directory! ")

    def transfer_file(self, old_path, new_path):
        if old_path.endswith(".py"):
            log_info(
                self.logger, "Start convert file: {} --> {}".format(old_path, new_path)
            )
            with open(old_path, "r", encoding="UTF-8") as f:
                code = f.read()
                if self.calculate_speed:
                    self.line_count += len(code.splitlines())
                root = ast.parse(code)

            self.transfer_node(root, old_path)
            code = astor.to_source(root)

            # format code
            if not self.no_format:
                try:
                    code = black.format_str(code, mode=black.Mode())
                except Exception as e:
                    log_info(
                        self.logger,
                        "Skip black format due to error: {}".format(str(e)),
                    )

                try:
                    code = isort.code(code)
                except Exception as e:
                    log_info(
                        self.logger,
                        "Skip isort format due to error: {}".format(str(e)),
                    )

                """
                try:
                    code = autoflake.fix_code(
                        code,
                        remove_all_unused_imports=False,
                        remove_unused_variables=True,
                        ignore_pass_statements=True,
                    )
                except Exception as e:
                    log_info(
                        self.logger,
                        "Skip autoflake format due to error: {}".format(str(e)),
                    )
                """
            if not self.only_complete:
                code = self.mark_unsupport(code, old_path)
            with open(new_path, "w", encoding="UTF-8") as file:
                file.write(code)
            log_info(
                self.logger, "Finish convert {} --> {}\n".format(old_path, new_path)
            )
        elif old_path.endswith("requirements.txt"):
            log_info(
                self.logger, "Start convert file: {} --> {}".format(old_path, new_path)
            )
            with open(old_path, "r", encoding="UTF-8") as old_file:
                code = old_file.read()
            code = code.replace("torch", "paddlepaddle-gpu")
            with open(new_path, "w", encoding="UTF-8") as new_file:
                new_file.write(code)
            log_info(
                self.logger, "Finish convert {} --> {}\n".format(old_path, new_path)
            )
        else:
            log_info(
                self.logger,
                "No need to convert, just Copy {} --> {}\n".format(old_path, new_path),
            )
            shutil.copyfile(old_path, new_path)

    def transfer_node(self, root, file):
        transformers = [
            ImportTransformer,  # import ast transformer
            TensorRequiresGradTransformer,  # attribute requires_grad transformer
            BasicTransformer,  # most of api transformer
            PreCustomOpTransformer,  # pre process for C++ custom op
            CustomOpTransformer,  # C++ custom op transformer
        ]
        for transformer in transformers:
            trans = transformer(
                root,
                file,
                self.imports_map,
                self.logger,
                self.all_api_map,
                self.unsupport_api_map,
            )
            trans.transform()
            self.torch_api_count += trans.torch_api_count
            self.success_api_count += trans.success_api_count
            if self.only_complete:
                break

    def mark_unsupport(self, code, file):
        lines = code.split("\n")
        mark_next_line = False
        in_str = False
        bracket_num = 0
        for i, line in enumerate(lines):
            if "Not Support auto convert" in line:
                mark_next_line = True
                continue
            else:
                # func decorator_list: @
                if mark_next_line and line != "@":
                    lines[i] = ">>>>>>" + line
                    mark_next_line = False
                    continue

            # """ torch.* """
            # " torch.* "
            # ' torch.* '
            # " (torch "
            # " torch) "
            # just remove the str, avoid str influence the torch api recognize
            rm_str_line = re.sub(r"[\"]{3}[^\"]+[\"]{3}", "", line)
            rm_str_line = re.sub(r"[\"]{1}[^\"]+[\"]{1}", "", rm_str_line)
            rm_str_line = re.sub(r"[\']{1}[^\']+[\']{1}", "", rm_str_line)

            # """
            # torch.*
            # """
            last_in_str = in_str
            if rm_str_line.count('"""') % 2 != 0:
                in_str = not in_str
            if last_in_str or in_str:
                continue

            # paddle.add(paddleformers.
            #   transformers.BertTokenizer)
            """
            # may be removed in future
            last_bracket_num = bracket_num
            bracket_num += rm_str_line.count("(")
            bracket_num -= rm_str_line.count(")")
            if last_bracket_num > 0:
                continue
            """

            for torch_package in self.imports_map[file]["torch_packages"]:
                if rm_str_line.startswith("%s." % torch_package):
                    lines[i] = ">>>>>>" + line
                    break

                # model_torch.npy
                # modeltorch.npy
                # 1torch.npy
                # paddleformers.transformers.*
                if re.match(r".*[^\w\.]{1}%s\." % torch_package, rm_str_line):
                    lines[i] = ">>>>>>" + line

        return "\n".join(lines)
