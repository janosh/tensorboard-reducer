---
title: 'TensorBoard Reducer: A Python package for aggregating TensorBoard logs'
tags:
  - Python
  - machine learning
  - PyTorch
  - TensorBoard
  - performance analysis
  - ensemble models
authors:
  - name: Janosh Riebesell
    orcid: 0000-0001-5233-3462
    corresponding: true
    affiliation: "1, 2"
affiliations:
 - name: Cavendish Laboratory, University of Cambridge, UK
   index: 1
 - name: Lawrence Berkeley Lab, Berkeley, USA
   index: 2
date: 2022-08-21
bibliography: [references.bib]
---

## Summary

TensorBoard Reducer is a `pip`-installable package to compute statistics (`mean`, `std`, `min`, `max`, `median` or any other [`numpy` operation](https://numpy.org/doc/stable/reference/routines.statistics)) [@harris_array_2020] of multiple TensorBoard runs and export the results back to disk as either new TensorBoard logs or CSV / JSON / Excel files. It is aimed at ML researchers dealing with large numbers of training runs. These commonly arise when e.g.

1. training regular or deep [@lakshminarayanan_simple_2016] model ensembles,
2. measuring epistemic uncertainty, or
3. when developing new model architectures / techniques and the goal is to reduce noise in loss/accuracy/error curves and establish stronger statistical significance of potential performance improvements.

## Statement of need

Over the past decade, artificial neural networks have proven a very versatile and scalable optimization technique. An entire industry has sprung up around them in record time, not least thanks to the high-quality open-source tooling and infrastructure built partly by academics [@fey_fast_2019] and even more so by big tech (e.g. PyTorch by Meta [@paszke_pytorch_2019], TensorFlow [@abadi_tensorflow_2015] and more recently JAX [@bradbury_jax_2018] by Google. While the upfront investment of developer time and computational resources required to build such tools is immense, they make it possible for researchers from all fields of science to train and deploy large models without the need to become or hire full-time engineers themselves, leading to high profile breakthroughs in several domain problems such as protein folding [@jumper_highly_2021] and solving the fractional electron problem [@kirkpatrick_pushing_2021] in DFT.

However, some aspects surrounding the application of these models have yet to catch up to their meteoric rise. Two we want to highlight are aggregate analysis and uncertainty estimation.

While tools like TensorBoard [@abadi_tensorflow_2015] and [Weights and Biases](https://wandb.ai) [@biewald_weights_2020] and others go a long way towards addressing the monitoring and debugging of NNs training as well as inspecting their output, [they don't allow aggregation of multiple runs](https://stackoverflow.com/q/43068200). Similarly, uncertainty estimation remains a largely unsolved problem [@kendall_what_2017] in neural networks. Yet to use them effectively and deploy into real-world applications, knowing when their predictions can be trusted and when to be wary is essential. While research into Bayesian deep learning is ongoing and may one day become the go-to solution, end-to-end learning of high-dimensional probability distributions over model weights remains a fragile and costly technique to this day. The equally expensive yet conceptually much simpler baseline technique of ensemble models remains hard to beat with approximate Bayesian methods [@lakshminarayanan_simple_2016]. In this setting one trains multiple independent copies of a model from different random initializations to garner a few glimpses into different regions of the loss landscape. The mean prediction of such ensembles tends to outperform each individual model's accuracy and also yields an epistemic uncertainty estimate from the variance across single-model predictions.

We believe the extra work of training model ensembles should be more prevalent across most regions of ML (with the exception of large language models due to the immense training cost) and offer TensorBoard Reducer as a tool to make the analysis of such ensembles easier.

![Mean and standard deviation computed using `tensorboard-reducer` and exported back to TensorBoard event files for the loss and accuracy curves of an ensemble model consisting of 5 `functorch` MLPs trained in parallel (see [`functorch_mlp_ensemble`](https://github.com/janosh/tensorboard-reducer/blob/main/examples/functorch_mlp_ensemble.ipynb)).\label{fig:functorch-ensemble-example}](../assets/2022-08-05-functorch-ensemble-landscape.png)

## Features and Application

Specifically, it designed to make the process of aggregating the results of related training runs fast and flexible. Built on top of Numpy and Pandas, it is well-integrated into the [NumFOCUS](https://numfocus.org) stack, supporting many aggregation operations such as `mean`, `std`, `min`, `max`, `median` (see [`numpy.statistics`](https://numpy.org/doc/stable/reference/routines.statistics)) and data export options such new TensorBoard event files as well as CSV / JSON / Excel data files (see [`pandas.io`](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html)) which can easily be extended should the need arise. It has comprehensive test coverage (98% at time of writing), doc string for all of the public API as well as 3 example notebooks demonstrating various use cases that can be launched in [Binder](https://github.com/jupyterhub/binderhub) with a single click.

|                                  |                                                                                                                                                                                                                                                                                      |                                                                                                                                                                                                   |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Basic Python API Demo**        | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/janosh/tensorboard-reducer/main?labpath=examples%2Fbasic_python_api_example.ipynb) <br>[View on GitHub](https://github.com/janosh/tensorboard-reducer/blob/main/examples/basic_python_api_example.ipynb) | Demonstrates how to work with local TensorBoard event files.                                                                                                                                      |
| **Functorch MLP Ensemble**       | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/janosh/tensorboard-reducer/main?labpath=examples%2Ffunctorch_mlp_ensemble.ipynb) <br>[View on GitHub](https://github.com/janosh/tensorboard-reducer/blob/main/examples/functorch_mlp_ensemble.ipynb)     | Shows how to aggregate run metrics with TensorBoard Reducer when training model ensembles using `functorch`.                                                                                      |
| **Weights & Biases Integration** | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/janosh/tensorboard-reducer/main?labpath=examples%2Fwandb_integration.ipynb) <br>[View on GitHub](https://github.com/janosh/tensorboard-reducer/blob/main/examples/wandb_integration.ipynb)               | Trains PyTorch CNN ensemble on MNIST, logs results to [WandB](https://wandb.ai), downloads metrics from multiple WandB runs, aggregates using `tb-reducer`, then re-uploads to WandB as new runs. |



## Acknowledgements

JR acknowledges support by the German Academic Scholarship Foundation (Studienstiftung). JR would also like to thank all [contributors that reported bugs and suggested features](https://github.com/janosh/tensorboard-reducer/issues?q=is:issue+is:closed).

## References
