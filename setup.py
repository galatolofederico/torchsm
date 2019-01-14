import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchsm",
    version="0.1.0",
    author="Federico A. Galatolo",
    author_email="federico.galatolo@ing.unipi.it",
    description="Pytorch implementation of the Stigmergic Memory",
    url="https://github.com/galatolofederico/torchsm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch==0.4.1",
        "numpy==1.15.2"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
        "Development Status :: 4 - Beta"
    ],
)