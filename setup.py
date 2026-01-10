from setuptools import find_packages, setup

setup(
    name="grok",
    packages=find_packages(),
    version="0.0.1",
    install_requires=[
        "blobfile",
        "numpy",
        "torch",
        "tqdm",
        "scipy",
        "mod",
        "matplotlib",
        "pandas"
    ],
)
