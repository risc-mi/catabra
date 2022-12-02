# Out-of-Distribution Detection (OOD)

## Configuration

Configuration for the command line tools should be done in the following format: 


    "ood_class": "autoencoder",
    "ood_source": "internal",
    "ood_kwargs": {}

* `ood_source` defines the origin of the detector. The following values are accepted:
    * `internal`: detector implemented directly in CaTabRa.
    * `pyod`: detector from PyOD library
    * `external`: detector implemented by an outside module

* `ood_class` is the name/path of an OOD detector:
        * if `ood_source` is `internal`: name of one of the modules in catabra.ood.internal (e.g. `soft_brownian_offset`)
        * if `ood_source` is `pyod`: name of one of the modules in pyod.models (e.g. `kde`)
        * if `ood_source` is `external`: full import path consisting of modules and class (e.g. `custom.module.CustomOOD`)
        * if value is <None> no OOD detection is performed

* `ood_kwargs` is a dictionary of optional parameters for specific OOD detectors in the form {"parameter-name": value, ...}.
e.g. for the autoencoder `{"target_dim_factor": 0.25, "reduction_factor": 0.9}`


## Classes

### Internal 

#### SoftBrownianOffset

Based on: *F. Möller et. al: ‘Out-of-distribution Detection and Generation using Soft Brownian Offset Sampling and Autoencoders’,
arXiv:2105.02965 [cs], May 2021, Accessed: Apr. 06, 2022. [Online]. Available: http://arxiv.org/abs/2105.02965*

Generates synthetic out-of-distribution samples by selecting a sample from the original dataset and transforming it
iteratively until the point’s minimum distance transgresses a set boundary. These samples are combined with the original+
data set in oder to train a classifier to differentiate between in-distribution and out-of-distribution samples.

#### Autoencoder 

An autoencoder is a neural network that consists of an encoder a decoder part. The encoder learns to reduce the input
data to a lower-dimensional space. The decoder learns to reconstruct the original points from the compressed data.
In the context of out-of-distribution detection it is assumed that in-distribution data results in a lower reconstruction
error then out-of-distribution data. A sample can be defined as OOD if the reconstruction error is above a certain threshold.
Refer to: https://www.tensorflow.org/tutorials/generative/autoencoder for an example application

### PyOD

[PyOD](https://pyod.readthedocs.io/) is a Python library for anomaly detection which includes well-established
algorithms. They are generally considered if there are at least two years since publication, 50+ citations, and usefulness
is probable.

Many anomaly detection algorithms can double as OOD detectors if the percentage of outliers is set to be very low. 
Examples of such detectors are:
* Isolation Forests (iforest)
* Kernel Density Estimation (kde)
* k-Nearest-Neighbours (knn)