import os
import subprocess
import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 8):
    raise RuntimeError(
        "PaConvert use new AST syntax and only supports Python version >= 3.8 now.")

with open('requirements.txt') as f:
    REQUIREMENTS = f.read().splitlines()

with open("README.md", "r")as f:
    LONG_DESCRIPTION = f.read()


packages = [
    'paconvert',
    'paconvert.transformer',
]

package_data = {
    'paconvert' : ['api_mapping.json', 'attribute_mapping.json']
}

def get_tag():
    try:
        cmd = ['git', 'describe', '--tags', '--abbrev=0', '--always']
        git_tag = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0].strip()
        git_tag = git_tag.decode()
    except:
        git_tag = '0.0.0'

setup(
    name='paconvert',
    version=get_tag(),
    author='PaddlePaddle',
    keywords=('code convert', 'paddle', 'paddlepaddle'),
    url='https://github.com/PaddlePaddle/PaConvert.git',
    packages = packages,
    package_data = package_data,
    install_requires=REQUIREMENTS,
    description='Code Convert to PaddlePaddle Toolkit',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    python_requires=">=3.8",
    setup_requires=['wheel'],
    entry_points={
        'console_scripts': [
            'paconvert=paconvert.main:main',
            ]
        }
    )
