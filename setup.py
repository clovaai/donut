"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import os
from setuptools import find_packages, setup

ROOT = os.path.abspath(os.path.dirname(__file__))


def read_version():
    data = {}
    path = os.path.join(ROOT, "donut", "_version.py")
    with open(path, "r", encoding="utf-8") as f:
        exec(f.read(), data)
    return data["__version__"]


def read_long_description():
    path = os.path.join(ROOT, "README.md")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


setup(
    name="donut-python",
    version=read_version(),
    description="OCR-free Document Understanding Transformer",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, Seunghyun Park",
    author_email="gwkim.rsrch@gmail.com",
    url="https://github.com/clovaai/donut",
    license="MIT",
    packages=find_packages(
        exclude=[
            "config",
            "dataset",
            "misc",
            "result",
            "synthdog",
            "app.py",
            "lightning_module.py",
            "README.md",
            "train.py",
            "test.py",
        ]
    ),
    python_requires=">=3.7",
    install_requires=[
        "transformers>=4.11.3",
        "timm",
        "datasets[vision]",
        "pytorch-lightning>=1.6.4",
        "nltk",
        "sentencepiece",
        "zss",
        "sconf>=0.2.3",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
