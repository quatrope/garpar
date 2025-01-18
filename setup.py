# This file is part of the
#   Garpar Project (https://github.com/quatrope/garpar).
# Copyright (c) 2021, 2022, 2023, 2024, Diego Gimenez, Nadia Luczywo,
# Juan Cabral and QuatroPe
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
    "attrs==23.2.0",
    "build==1.2.1",
    "clarabel==0.8.1",
    "contourpy==1.2.1",
    "custom-inherit==2.4.1",
    "cvxpy==1.6.0",
    "cycler==0.12.1",
    "Deprecated==1.2.14",
    "ecos==2.0.13",
    "fonttools==4.52.4",
    "h5py==3.11.0",
    "importlib_metadata==7.1.0",
    "Jinja2==3.1.4",
    "joblib==1.4.2",
    "kiwisolver==1.4.5",
    "MarkupSafe==2.1.5",
    "matplotlib==3.8.4",
    "methodtools==0.4.7",
    "numpy==1.26.4",
    "osqp==0.6.7",
    "packaging==24.1",
    "pandas==2.2.3",
    "pillow==10.4.0",
    "PuLP==2.8.0",
    "pyparsing==3.1.2",
    "pyportfolioopt==1.5.5",
    "pyproject_hooks==1.1.0",
    "python-dateutil==2.9.0.post0",
    "pytz==2024.1",
    "qdldl==0.1.7.post2",
    "scikit-learn==1.6.0",
    "scipy==1.14.1",
    "scs==3.2.4.post1",
    "seaborn==0.13.2",
    "setuptools==69.5.1",
    "six==1.16.0",
    "threadpoolctl==3.5.0",
    "tox==4.15.0",
    "tzdata==2024.1",
    "wirerope==0.4.7",
    "wrapt==1.16.0",
    "zipp==3.19.0",
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
    description="""Generation and Analysis of Artificial and Real
                   Portfolio Returns""",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="""Diego N Gimenez Irusta, Nadia Luczywo, Juan B Cabral &
              Trinchi Chalela""",
    author_email="diego.gimenez@unc.edu.ar",
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
    include_package_data=True,
)
