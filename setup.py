from setuptools import setup, find_packages

setup(
    name="emotune",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'dtaidistance',
    ],
    python_requires='>=3.6',
)
