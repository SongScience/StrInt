from setuptools import setup
from setuptools import find_packages

setup( name = 'pyStrint',
version = '0.0.30',
description='Deciphering more accurate cell-cell interactions by modeling cells and their interactions',
url='https://github.com/deepomicslab/StrInt',
author='Jingwan WANG',
author_email='wanwang6-c@my.cityu.edu.hk',
license='MIT',
packages=find_packages(),
install_requires = [
    'anndata==0.10.5.post1',
    'numpy==1.22.3',
    'pandas==1.5.2',
    'scipy==1.9.3',
    'scanpy==1.9.8',
    'scikit-learn==1.6.1',
    'umap-learn==0.5.2',
    'loess==2.1.2',
    'smurf-imputation==1.0.9',
    'matplotlib==3.8.4',
    'seaborn==0.13.2',
    'matplotlib-venn==0.11.10',
    'configargparse==1.7',
    'plotly==5.24.1',
    'kaleido==0.2.1',
    'nbformat==5.10.4'
],
package_data={
    'pyStrint': [
        'LR/*.txt',
        'pipelines/*',
    ],
},
include_package_data=True
)
