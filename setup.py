from setuptools import setup, find_packages
  
setup(
    name='mvit',
    version='0.1',
    description='Multi-stride History-aware Transformer for surgical phase classification',
    author='Jobin Jose',
    author_email='jobin.jbn@gmail.com',
    packages=find_packages(),
    install_requires=['av'],
)
