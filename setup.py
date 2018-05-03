try:
    from setuptools import setup, find_packages
except ImportError:
    from disutils.core import setup, find_packages

import cnmfpy

config = {
    'name': 'cnmfpy',
    'description': 'Tools for Convolutive Non-Negative Matrix Factorization',
    'author': 'Anthony Degleris, Alex Williams',
    'author_email': 'degleris@stanford.edu',
    'url': 'https://github.com/degleris1/cnmfpy',
}

setup(**config)