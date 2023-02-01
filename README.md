# CaTabRa: Effective Analysis of Tabular Data

## Description

**CaTabRa** is a Python package for analyzing tabular data, in a largely automated way. This includes generating
descriptive statistics, creating out-of-distribution detectors, training prediction models for classification and
regression tasks, and evaluating/explaining/applying these models on unseen data.

CaTabRa is both a command-line tool and a library, which means it can be easily integrated into other projects.

## Installation

Install the relevant packages, as listed in `env.yml` and `requirements.txt`, in a new or existing environment. The
key requirements are Python 3.9, [pandas](https://pandas.pydata.org/) and
[auto-sklearn](https://automl.github.io/auto-sklearn/master/).

**IMPORTANT**: auto-sklearn currently only runs under Linux, which means that CaTabRa only runs under Linux, too. If
on Windows, you can use a virtual machine, like [WSL 2](https://docs.microsoft.com/en-us/windows/wsl/about).

## Code Overview

### Naming Conventions

tbd

### Notable Modules

* `__main__.py`
  * Implementation of the command-line interface.
  * Calls functions implemented in other modules.
* `analysis/__init__.py`
  * Implementation of analysis functionality, in function `analyze()`.
  * Prepares data, creates output directory, scans config, calls AutoML backend.
* `evaluation/__init__.py`
  * Implementation of evaluation functionality, in function `evaluate()`.
  * Prepares data, creates output directory, applies models, generates statistics and plots.
* `automl/base.py`
  * Abstract interface for AutoML backends.
  * Resembles the `BaseEstimator` interface of scikit-learn.
* `base/config.py`
  * Default configuration.
* `automl/askl/__init__.py`
  * Implementation of the auto-sklearn backend.
  * Necessary packages are only imported if this backend is actually used; other backends should follow a similar
      pattern.
* `ood/base.py`
  * Abstract base class for out-of-distribution (OOD) detection.
  * Contains factory method for creation of sub-classes.
* `ood/internal/`
  * Folder for OOD detectors implemented in CaTaBra.
* `ood/internal/`
  * Folder for OOD detectors implemented in CaTaBra.
* `ood/pyod.py`
  * Wrapper class for making PyOD detectors confirm with the base class. 
  * [PyOD](https://pyod.readthedocs.io/en/latest/) is a library for anonmaly detection
* `util/encofing.py`
  * Implementation of the feature encoding functionality.
  * Responsible for converting time-like attributes into numeric attributes, among other things.
  * Implements the scikit-learn `BaseEstimator` interface, with methods `fit()`, `transform()` and
      `inverse_transform()`.
* `util/io.py`
  * Utility I/O functions, for tables/DataFrames, JSON serialization, and pickling.
* `util/logging.py`
  * Utility functions for printing messages and prompting the user for input.
  * Functionality for mirroring `print()`-statements to a log file.