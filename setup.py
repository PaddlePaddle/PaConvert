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

import subprocess
import sys

from setuptools import setup

if sys.version_info < (3, 8):
    raise RuntimeError(
        "PaConvert use new AST syntax and only supports Python version >= 3.8 now."
    )

with open("requirements.txt", "r") as f:
    REQUIREMENTS = f.read().splitlines()

with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()


packages = [
    "paconvert",
    "paconvert.transformer",
]

package_data = {
    "paconvert": [
        "api_mapping.json",
        "api_wildcard_mapping.json",
        "attribute_mapping.json",
        "api_alias_mapping.json",
    ]
}


def get_tag():
    try:
        cmd = ["git", "tag"]
        git_tag = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0].strip()
        git_tag = git_tag.decode()
        # keep latest Paddle version
        git_tag = git_tag.split("\n")[-1]
    except:
        git_tag = "0.0.0"

    if not git_tag:
        git_tag = "0.0.0"

    return git_tag


with open("paconvert/version.py", "w") as f:
    content = f"""# THIS FILE IS GENERATED FROM PACONVERT SETUP.PY
__version__     = '{get_tag()}'
"""
    f.write(content)

setup(
    name="paconvert",
    version=get_tag(),
    description="PaddlePaddle Code Convert Toolkits",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author_email="Paddle-better@baidu.com",
    maintainer="PaddlePaddle",
    maintainer_email="Paddle-better@baidu.com",
    url="https://github.com/PaddlePaddle/PaConvert",
    download_url="https://github.com/PaddlePaddle/PaConvert",
    license="Apache Software License",
    python_requires=">=3.8",
    install_requires=REQUIREMENTS,
    setup_requires=["wheel"],
    packages=packages,
    package_data=package_data,
    entry_points={
        "console_scripts": [
            "paconvert=paconvert.main:main",
        ]
    },
)
