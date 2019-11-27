from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="feature_elimination",
    version="0.1",
    author="Kynon JM Benjamin and Apua CM Paquola",
    author_email="kj.benjamin90@gmail.com",
    decription="A package to preform feature elimination for random forest",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KrotosBenjamin/erwin_paquola/tree/master/feature_elimination_pkg",
    packages=find_packages('src'),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPLv3 License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
