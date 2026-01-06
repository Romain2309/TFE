#!/usr/bin/env python3
"""
Setup script for Gravitational Wave Sky Localization package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / 'README.md'
long_description = readme_file.read_text() if readme_file.exists() else ''

setup(
    name='gw-skylocalization',
    version='1.0.0',
    author='Master Thesis Project',
    description='Deep learning for gravitational wave sky localization',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/gw-skylocalization',
    packages=find_packages(exclude=['tests', 'scripts', 'notebooks', 'docs']),
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.10.0',
        'torchvision>=0.11.0',
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'h5py>=3.0.0',
        'healpy>=1.15.0',
        'matplotlib>=3.3.0',
        'tqdm>=4.60.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.12',
            'black>=21.0',
            'flake8>=3.9',
            'mypy>=0.910',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme>=0.5',
        ],
    },
    entry_points={
        'console_scripts': [
            'gw-train=scripts.train:main',
            'gw-evaluate=scripts.evaluate:main',
            'gw-compare=scripts.compare_models:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='gravitational-waves deep-learning astronomy localization neural-networks',
)
