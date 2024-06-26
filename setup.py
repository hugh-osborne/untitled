# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='untitled',
    version='0.0.1',
    description='A human biophysical model',
    long_description=readme,
    author='Hugh Osborne',
    author_email='hugh.osborne@gmail.com',
    url='https://github.com/hugh-osborne/untitled',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

