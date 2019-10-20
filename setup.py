from setuptools import setup, find_packages

setup(
    name='sngan_projection',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'chainer',
        'cython',
        'numpy',
        'tqdm',
        'pillow',
        'pyyaml'
    ]
)
