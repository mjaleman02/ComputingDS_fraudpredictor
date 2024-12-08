from setuptools import setup, find_packages

setup(
    name='fraud_predictor',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'lightgbm',
        'matplotlib',
        'pytest',
        'pytest-cov',
    ],
)
