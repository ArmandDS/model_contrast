from setuptools import setup, find_packages
from os.path import abspath, dirname, join

README_MD = open(join(dirname(abspath(__file__)), "README.md")).read()

setup(
    name='model contrast',
    version='0.1.3',
    description='Compare two ML models.',
    long_description=README_MD,
    long_description_content_type="text/markdown",
    author='Armand Olivares',
    author_email='armandds@users.noreply.github.com',
    license='MIT',
    url='https://github.com/ArmandDS/model_contrast',
    packages=find_packages(exclude="tests"),
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.1.5',
        'statsmodels>=0.10.2',
        'scipy>=1.4.1',
        'scikit-learn>=0.22.2'
    ],
)