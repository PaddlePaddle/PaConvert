import os
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    REQUIREMENTS = f.read().splitlines()

with open("README.md", "r")as f:
    LONG_DESCRIPTION = f.read()


packages = [
    'paddleconverter',
    'paddleconverter.transformer',
    'paddleconverter.tests'
]

package_data = {
    'paddleconverter' : ['api_mapping.json']
}

setup(
    name='paddleconverter',
    version='1.0',
    author='PaddlePaddle',
    keywords=('paddleconverter tool', 'paddle', 'paddlepaddle'),
    url='https://github.com/zhouwei25/paddleconverter',
    packages = packages,
    package_data = package_data,
    install_requires=REQUIREMENTS,
    description='Convert python project from torch-1.8 to paddle-2.4',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    python_requires=">=3.7",
    setup_requires=['wheel'],
    entry_points={
        'console_scripts': [
            'paddleconverter=paddleconverter.main:main',
            ]
        }
    )
