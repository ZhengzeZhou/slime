# slime

This repository holds code for replicating experiments in papar [S-LIME: Stabilized-LIME for Model Explanation]() to appear in [KDD2021](https://www.kdd.org/kdd2021/). 

The repository is built on the implementation of [lime](https://github.com/marcotcr/lime) with added functionalities. It is still under development. 

## Introduction

It has been shown that post hoc explanations based on perturbations (such as LIME) exhibit large instability, posing serious challenges to the effectiveness of the method itself and harming user trust. S-LIME stands for Stabilized-LIME, which utilizes a hypothesis testing framework based on central limit theorem for determining the number of perturbation points needed to guarantee stability of the resulting explanation. 

## Installation

clone the repository and run:

```sh
pip install .
```

## Usage

Currently, S-LIME only support tabular data and when feature selection method is set to "lasso_path".

The following screenshot shows a typical usage of LIME using breasd cancer data. 

![demo1](doc/images/demo1.png)
