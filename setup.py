from setuptools import setup, find_packages

setup(
    name='mlproject',
    version='0.0.1',
    author='Katherine Graham',
    author_email='katherinelgraham@gmail.com',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines()
)