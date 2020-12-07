# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE.md') as f:
    license = f.read()

setup(
    name='XGBstrc',
    version='0.1.0',
    description='Node classification model',
    long_description=readme,
    author='Miguel Romero',
    author_email='romeromiguelin@gmail.com',
    url='https://github.com/omicas/P5.git/miguel/xgbstrc',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
