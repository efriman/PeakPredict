from setuptools import setup
import os
from os import path
import io

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def _read(*parts, **kwargs):
    filepath = os.path.join(os.path.dirname(__file__), *parts)
    encoding = kwargs.pop("encoding", "utf-8")
    with io.open(filepath, encoding=encoding) as fh:
        text = fh.read()
    return text


def get_requirements(path):
    content = _read(path)
    return [
        req
        for req in content.split("\n")
        if req != "" and not (req.startswith("#") or req.startswith("-"))
    ]


setup_requires = []

INSTALL_REQUIRES = get_requirements("requirements.txt")

setup(
    name="overlap_peak_tables",
    packages=["overlap_peak_tables", "overlap_peak_tables.lib"],
    entry_points={
        "console_scripts": [
            "overlap_peak_tables = overlap_peak_tables.overlap_peak_tables_CLI:main",
            "predict_features = overlap_peak_tables.predict_features_CLI:main",
        ]
    },
    setup_requires=setup_requires,
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3.8",
    description="Simple script to count the number of overlapping or closest features between one and multiple bed files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Source": "https://github.com/efriman/overlap_peak_tables",
        "Issues": "https://github.com/efriman/overlap_peak_tables/issues",
    },
    author="Elias Friman",
    author_email="elias.friman@gmail.com",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
