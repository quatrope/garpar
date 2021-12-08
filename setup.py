# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
# License: MIT
#   Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================

from setuptools import find_packages, setup

# =============================================================================
# CONSTANTS
# =============================================================================

with open("README.md", "r") as fp:
    LONG_DESCRIPTION = fp.read()

REQUIREMENTS = ["numpy", "pandas", "scipy", "seaborn", "matplotlib"]

# =============================================================================
# SETUP
# =============================================================================

setup(
    name="garpar",
    version="0.2",
    description="Market generation and portfolio analysis",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Nadia Luczywo",
    author_email="nluczywo@gmail.com",
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
