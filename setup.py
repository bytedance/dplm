
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


from setuptools import find_packages, setup


setup(
    name="ByProt",  
    version="1.0.0",
    description="A pytorch library for swift protein design research and developing.",
    author="ByteDance Research",
    author_email="zhengzaixiang@bytedance.com",
    # url="https://github.com/bytedance/ByProt", 
    install_requires=open("requirements.txt").readlines(),
    package_dir={"": "src"},
    packages=find_packages("src")
)
