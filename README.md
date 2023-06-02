# CaTabRa

<p align="center">
  <a href="#About"><b>About</b></a> &bull;
  <a href="#Quickstart"><b>Quickstart</b></a> &bull;
  <a href="#Examples"><b>Examples</b></a> &bull;
  <a href="#Documentation"><b>Documentation</b></a> &bull;
  <a href="#References"><b>References</b></a> &bull;
  <a href="#Contact"><b>Contact</b></a> &bull;
  <a href="#Acknowledgments"><b>Acknowledgments</b></a>
</p>

<div align="center">
  <img src="https://raw.githubusercontent.com/risc-mi/catabra/main/doc/figures/workflow.png" width="700px" />
</div>

[![Platform Support](https://img.shields.io/badge/platform-Linux-blue)]()

## About

**CaTabRa** is a Python package for analyzing tabular data in a largely automated way. This includes generating
descriptive statistics, creating out-of-distribution detectors, training prediction models for classification and
regression tasks, and evaluating/explaining/applying these models on unseen data.

CaTabRa is both a command-line tool and a library, which means it can be easily integrated into other projects.

## Quickstart

### Installation

Clone the repository and install the package with [Poetry](https://python-poetry.org/docs/).
Set up a new Python environment with Python >=3.9 (e.g. using
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)),
activate it, and then run

```shell
git clone https://github.com/risc-mi/catabra.git
cd catabra
poetry install
```

The project is installed in editable mode by default. This is useful if you plan to make changes to CaTabRa's code.

**IMPORTANT**: CaTabRa currently only runs on Linux, because
[auto-sklearn only runs on Linux](https://automl.github.io/auto-sklearn/master/installation.html). If on Windows,
you can use a virtual machine, like [WSL 2](https://docs.microsoft.com/en-us/windows/wsl/about), and install CaTabRa
there. If you want to use Jupyter, install Jupyter on the virtual machine as well and launch it with the `--no-browser`
flag.

### Usage Mode 1: Command-Line

```shell
python -m catabra analyze example_data/breast_cancer.csv --classify diagnosis --split train --out breast_cancer_result
```

This command analyzes `breast_cancer.csv` and trains a prediction model for classifying the samples according to column
`"diagnosis"`. Column `"train"` is used for splitting the data into a train- and a test set, which means that the final
model is automatically evaluated on the test set after training. All results are saved in directory `breast_cancer_out`.

```shell
python -m catabra explain breast_cancer_result --on example_data/breast_cancer.csv --out breast_cancer_result/expl
```

This command explains the classifier trained in the previous command by computing SHAP feature importance scores for
every sample. The results are saved in directory `breast_cancer_result/expl`. Depending on the type of the trained
models, this command may take several minutes to complete.

### Usage Mode 2: Python

The two commands above translate to the following Python code:

```python
from catabra.analysis import analyze
from catabra.explanation import explain

analyze("example_data/breast_cancer.csv", classify="diagnosis", split="train", out="breast_cancer_result")
explain("example_data/breast_cancer.csv", "breast_cancer_result", out="breast_cancer_result/expl")
```

### Results

Invoking the two commands generates a bunch of results, most notably

* the trained classifier
* descriptive statistics of the underlying data<br>
  <img src="https://raw.githubusercontent.com/risc-mi/catabra/main/doc/figures/breast_cancer_descriptive.png" width="600px" />
* performance metrics in tabular and graphical form<br>
  <img src="https://raw.githubusercontent.com/risc-mi/catabra/main/doc/figures/breast_cancer_confusion_matrix.png" width="200px" />
  <img src="https://raw.githubusercontent.com/risc-mi/catabra/main/doc/figures/breast_cancer_threshold_metric.png" width="400px" />
* feature importance scores in tabular and graphical form<br>
  <img src="https://raw.githubusercontent.com/risc-mi/catabra/main/doc/figures/breast_cancer_beeswarm.png" width="600px" />
* ... and many more.

## Examples

The source notebooks for all our examples can be found in the
[examples folder](https://github.com/risc-mi/catabra/tree/main/examples/).

### Walk-Through Tutorials

* **[Workflow.ipynb](https://catabra.readthedocs.io/en/latest/jupyter/workflow.html)**
  * Analyze data with a binary target
  * Train a high-quality classifier with automatic model selection and hyperparameter tuning
  * Investigate the final classifier and the training history
  * Calibrate the classifier on dedicated calibration data
  * Evaluate the classifier on held-out test data
  * Explain the classifier by computing SHAP- and permutation importance scores
  * Apply the classifier to new samples
* **[Longitudinal.ipynb](https://catabra.readthedocs.io/en/latest/jupyter/longitudinal.html)**
  * Process longitudinal data by resampling into "samples x features" format

### Short Examples

* **[Prediction-Tasks.ipynb](https://catabra.readthedocs.io/en/latest/jupyter/prediction_tasks.html)**
  * Binary classification
  * Multiclass classification
  * Multilabel classification
  * Regression
* **[House-Sales-Regression.ipynb](https://catabra.readthedocs.io/en/latest/jupyter/house_sales_regression.html)**
  * Predicting house prices
* **[Performance-Metrics.ipynb](https://catabra.readthedocs.io/en/latest/jupyter/performance_metrics.html)**
  * Change hyperparameter optimization objective
  * Specify metrics to calculate during model training
* **[Plotting.ipynb](https://catabra.readthedocs.io/en/latest/jupyter/plotting.html)**
  * Create plots in Python
  * Create interactive plots
* **[AutoML-Config.ipynb](https://catabra.readthedocs.io/en/latest/jupyter/automl_config.html)**
  * General configuration
    * Ensemble size
    * Time- and Memory budget
    * Number of parallel jobs
  * Auto-Sklearn-specific configuration
    * Model classes and preprocessing steps
    * Resampling strategies for internal validation
    * Grouped splitting
* **[Fixed-Pipeline.ipynb](https://catabra.readthedocs.io/en/latest/jupyter/fixed_pipeline.html)**
  * Specify fixed ML pipeline (no automatic hyperparameter optimization)
  * Manually configure hyperparameters
  * Suitable for creating baseline models

### Extending CaTabRa

* **[AutoML-Extension.ipynb](https://catabra.readthedocs.io/en/latest/jupyter/automl_extension.html)**
  * Add new AutoML backend
* **[Explanation-Extension.ipynb](https://catabra.readthedocs.io/en/latest/jupyter/explanation_extension.html)**
  * Add new explanation backend
* **[OOD-Extension.ipynb](https://catabra.readthedocs.io/en/latest/jupyter/ood_extension.html)**
  * Add new OOD detection backend

## Documentation

API Documentation as well as detailed documentation for a couple of specific aspects of CaTabRa,
like its  command-line interface, available performance metrics, built-in OOD-detectors and model explanation details
can be found on our [ReadTheDocs](https://catabra.readthedocs.io/en/latest/index.html).


## References

If you use CaTabRa in your research, we would appreciate citing the following conference paper:

* A. Maletzky, S. Kaltenleithner, P. Moser and M. Giretzlehner.
  *CaTabRa: Efficient Analysis and Predictive Modeling of Tabular Data*. In: I. Maglogiannis, L. Iliadis, J. MacIntyre
  and M. Dominguez (eds), Artificial Intelligence Applications and Innovations (AIAI 2023). IFIP Advances in
  Information and Communication Technology, vol 676, pp 57-68, 2023.
  [DOI:10.1007/978-3-031-34107-6_5](https://doi.org/10.1007/978-3-031-34107-6_5)

  ```
  @inproceedings{CaTabRa2023,
    author = {Maletzky, Alexander and Kaltenleithner, Sophie and Moser, Philipp and Giretzlehner, Michael},
    editor = {Maglogiannis, Ilias and Iliadis, Lazaros and MacIntyre, John and Dominguez, Manuel},
    title = {{CaTabRa}: Efficient Analysis and Predictive Modeling of Tabular Data},
    booktitle = {Artificial Intelligence Applications and Innovations},
    year = {2023},
    publisher = {Springer Nature Switzerland},
    address = {Cham},
    pages = {57--68},
    isbn = {978-3-031-34107-6},
    doi = {10.1007/978-3-031-34107-6_5}
  }
  ```

## Contact

If you have any inquiries, please open a GitHub issue.

## Acknowledgments

This project is financed by research subsidies granted by the government of Upper Austria. RISC Software GmbH is Member
of UAR (Upper Austrian Research) Innovation Network.