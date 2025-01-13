from setuptools import setup, find_packages

dependencies = [
    "torch",
    "torchvision",
    "Pillow",
    "matplotlib",
    "numpy",
    "datasets",
    "tensorboard"
]

setup(
    name="InfantVisualPerception",
    version="0.1",
    packages=find_packages(),
    install_requires=dependencies,
    include_package_data=True,
    python_requires=">=3.9",
)
