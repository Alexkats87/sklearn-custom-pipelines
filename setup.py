"""Setup configuration for sklearn-custom-pipelines."""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='sklearn-custom-pipelines',
    version='0.1.0',
    author='Alex Kats',
    author_email='your-email@example.com',
    description='A collection of custom scikit-learn transformers for machine learning pipelines',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Alexkats87/sklearn-custom-pipelines',
    packages=find_packages(exclude=['tests', '*.tests', '*.tests.*', 'tests.*']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    python_requires='>=3.7',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.10',
            'black>=21.0',
            'flake8>=3.9',
            'sphinx>=4.0',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
