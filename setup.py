# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================

import os

from setuptools import find_packages, setup

# =============================================================================
# CONSTANTS
# =============================================================================

REQUIREMENTS = [
    "numpy",
    "pandas",
    "scipy",
    "seaborn",
    "matplotlib",
    "h5py",
    "joblib",
    "PyPortfolioOpt",
    "pyquery",
    "scikit-learn",
]


with open("README.md", "r") as fp:
    LONG_DESCRIPTION = fp.read()


GARPAR_INIT_PATH = os.path.join("garpar", "__init__.py")

with open(GARPAR_INIT_PATH, "r") as f:
    for line in f:
        if line.startswith("__version__"):
            VERSION = line.split("=", 1)[-1].replace('"', "").strip()
            break


# =============================================================================
# SETUP
# =============================================================================

setup(
    name="garpar",
    version=VERSION,
    description="Market generation and portfolio analysis",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Nadia Luczywo, Juan B Cabral & Trinchi Chalela",
    author_email="jbcabral@unc.edu.ar",
    url="https://github.com/quatrope/garpar",
    license="The MIT License",
    install_requires=REQUIREMENTS,
    keywords=[
        "market simulation",
        "informational efficiency",
        "portfolio optimization",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
    ],
    packages=find_packages(".", include=["garpar*"]),
)
