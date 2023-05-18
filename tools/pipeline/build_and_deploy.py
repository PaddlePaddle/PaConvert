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

import sys

from setuptools import setup

if sys.version_info < (3, 8):
    raise RuntimeError(
        "PaConvert use new AST syntax and only supports Python version >= 3.8 now."
    )

with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()


packages = [
    "paconvert",
    "paconvert.transformer",
]

package_data = {"paconvert": ["api_mapping.json", "attribute_mapping.json"]}

setup(
    name="paconvert",
    version="0.0.0",
    author="PaddlePaddle",
    keywords=("code convert", "paddle", "paddlepaddle"),
    url="https://github.com/PaddlePaddle/PaConvert.git",
    packages=packages,
    package_data=package_data,
    install_requires=REQUIREMENTS,
    description="Code Convert to PaddlePaddle Toolkit",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    python_requires=">=3.8",
    setup_requires=["wheel"],
    entry_points={
        "console_scripts": [
            "paconvert=paconvert.main:main",
        ]
    },
)
