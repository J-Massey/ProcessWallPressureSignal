# setup.py at repository root
from setuptools import setup, find_packages

setup(
    name="pressureprocess",
    version="0.1.0",
    packages=find_packages(),              # Automatically find the package directory
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "icecream",
        "scienceplots",
        "torch",
    ],
    # optional: include_package_data=True, and MANIFEST.in to include data files if needed
)
