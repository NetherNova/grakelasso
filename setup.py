__author__ = 'martin'

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='sample',
    version='0.1',
    description='',
    long_description=long_description,

    url='https://github.com/nethernova/grakelasso',

    author='Martin Ringsquandl',
    author_email='martin.ringsquandl@gmail.com',

    license='',
)