# UQ4DD: Uncertainty Quantification for Drug Discovery

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10_%7C_3.11-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0%2B-792ee5?logo=lightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![wandb](https://img.shields.io/badge/Weights_%26_Biases-black?logo=weightsandbiases&logoColor=yellow)](https://wandb.ai/site)

[![openreview](https://img.shields.io/badge/Workshop_Paper-OpenReview.net-%23b31b1b)](https://openreview.net/forum?id=5B8tsitI5s)
[![arxiv](https://img.shields.io/badge/Preprint-arXiv:2409.04313-%23b31b1b)](https://arxiv.org/abs/2409.04313)


**[Installation](#installation)**
| **[Data](#data)**
| **[Usage](#usage)**
| **[Citation](#citation)**

This package provides the full methodology used in a collection of publication for uncertainty quantification during the prediction of properties of molecular compounds and drug-target interactions. While the published research was performed on proprietary data from AstraZeneca, this package runs on datasets from [Therapeutics Data Commons](https://tdcommons.ai/).


## Installation

The main requirements are,
- Python >= 3.8
- CUDA >= 11.7
- PyTorch >= 2.0
- Lightning >= 2.0

Additional packages include sklearn, hydra, RDKit, PyTDC.

Logging is supported with [Weights & Biases](https://wandb.ai/site).

Install full conda environment with,

```bash
$ conda env create -f environment.yaml
```

The code structure is based on the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template/).

## Data
While the main contributions of this work require proprietary internal pharmaceutical data with temporal information, standard deviation of experiments, and censored labels, the full source code is made available on public datasets from the [Therapeutics Data Commons](https://tdcommons.ai/). Specifically, we provide examples for the random splits of the following ADME tasks in

**Regression**
- Intrinsic Clearance Hepatocyte (AstraZeneca), including 1,020 drug compounds. 
- Intrinsic Clearance Microsome (AstraZeneca), including 1,102 drug compounds. 
- Plasma Protein Binding Rate (AstraZeneca), including 1,614 drug compounds.
- Lipophilicity (AstraZeneca), including 4,200 drug compounds.
- Solubility (AqSolDB), including 9,982 drug compounds.

**Classification**
- CYP P450 2C9 (Veith), including 12,092 drug compounds.
- CYP P450 3A4 (Veith), including 12,328 drug compounds.
 
However, the framework can easily be extended to all other single-instance prediction datasets from PyTDC as well as to other provided splitting strategies. Furthermore, the code is capable of handling temporal data splitting, probabilstic labels, and censored labels if the required information is provided by the user. 

## Usage

Single-instance drug-target interactions and molecular properties can be predicted together with the quantified uncertainty using ensemble-based, Gaussian, and Bayesian approaches seen in the Figure below.

![plot](https://github.com/MolecularAI/uq4dd/blob/main/illustrations/uq_method.png)

### Encoders
Currently, the supported encoder for drugs is limited to RDKit ECFP, with optional radius (default=2) and size (default=1024). However, we strongly encourage the use of state-of-the-art pre-trained encoders such as [CDDD](https://github.com/jrwnter/cddd) and [MolBERT](https://github.com/BenevolentAI/MolBERT).

### Learning frameworks
We provide three learning frameworks with uncertainty quantification, (Baselines) a wrapper around [scikit-learn](https://scikit-learn.org/stable/) models such as Random Forest (RF), (Deep Learning) regular fully-connected neural networks, and (Bayesian Learning) an approximation of a Bayesian neural network using Bayes by Backprop (Blundell, et al., 2015). 

**Baselines**

Primarily, the RF from scikit-learn can be used as a baseline of an ensemble-based approach where the variance over the decision trees is taken as an estimate of epistemic uncertainty. Furthermore, the code can be extended to other models from scikit-learn, such as Support Vector Machine (SVM) or Gaussian Process (GP), or to the XGBoost model from xgboost. Train and evaluate the baseline models using 

```bash
$ python uq4dd/train.py model=rf db=$DATASET
```

where $DATASET is the name of a config file for the given dataset, e.g. `lipo` for Lipophilicity.

**Deep Learning**

The framework for developing fully connected neural networks with abilities to quantify uncertainty supports the following three base estimators,

- Single output multi-layer perceptron (MLP), `model=mlp`,
- Mean-Variance Estimator (MVE), `model=mve`,
- Evidential model, `model=evidential`.

Each of these base estimators can be pre-trained any number of times such that their best checkpoints in terms of validation loss are saved for later use, with

```bash
$ python uq4dd/pretrain.py model=$MODEL db=$DATASET
```

The respective pre-trained base estimators can then be used to evaluate the following uncertainty quantification approaches,

- Ensemble (Lakshminarayanan et al., 2017), an ensemble of *k* MLPs where the mean prediction is taken as the final estimated property and the variance of the predictions is taken as an estimate of the epistemic uncertainty, `model=ensemble`.
- MC-Dropout (Gal and Ghahramani, 2016), an ensemble of sampled predictions from a single MLP with dropout applied during inference such that the mean prediction is taken as the final estimated property and the variance of the predictions is taken as an estimate of the epistemic uncertainty, `model=mc`.
- Gaussian (Nix and Weigend, 1994), a single MVE that directly predicts the property and an estimate of the aleatoric uncertainty, `model=mve`.
- Gaussian Ensemble (Lakshminarayanan et al., 2017), an ensemble of *k* MVEs such that the mean prediction is taken as the final estimated property, the variance of the predictions is taken as an estimate of the epistemic uncertainty, and the mean predicted aleatoric uncertainty estimates is taken as the final aleatoric uncertainty, `model=gmm`.
- Evidential (Amini et al., 2020), a single evidential model trained to predict the four parameters, $\gamma, \nu, \alpha, \beta$, such that the predicted property is $\gamma$, the aleatoric uncertainty estimate is $\frac{\beta}{\nu(\alpha - 1)}$, and the epistemic uncertainty estimate is $\frac{\beta}{\alpha - 1}$, `model=evidential`.

Once enough base estimators have been pre-trained the model can be evaluated with, 

```bash
$ python uq4dd/eval.py model=$MODEL db=$DATASET
```

**Bayesian Learning**

In the Bayesian framework, an approximation of a Bayesian Neural Network can be trained using Bayes by Backprop (Blundell, et al., 2015) and evaluated with

```bash
$ python uq4dd/pretrain.py model=bnn_train db=$DATASET
$ python uq4dd/eval.py model=bnn_eval db=$DATASET
```

### Calibration

Probability calibration can be achieved with Platt-scaling (Platt, 1999) and Venn-ABERs (Vovk and Petej, 2014) for classification applications. Compare both calibration approaches by specifying `model.recalibrate=platt_va` in any of the above scripts. 

For regression applications we support re-calibration of uncertainty estimates using a fitted linear error-based calibration (Rasmussen et al., 2023). Apply the re-calibration approach by specifying `model.recalibrate=uq_linear` in any of the above scripts.

### Evaluation
We evaluate performance using the respective loss functions, as well as any classification metrics, e.g. Accuracy, AUC. Probability calibration is evaluated with ECE, ACE, and Brier-score. Uncertainty Quantification is related to performance in the Expected Noramlized Calibration Error (ENCE) and the Gaussian Negative Log Likelihood (NLL). 

### Logging

Optionally, specify a configuration for a [Weights & Biases](https://wandb.ai/site) logger in `config/logger/` and run with flag `logger={name of yaml file}`. See example `config/logger/user.yaml`.

### Hyperparameter Optimization

The hyperparameters for a given model can be optimized on a given dataset using the [Weights & Biases](https://wandb.ai/site) sweep as follows,

```bash
$ python uq4dd/sweep.py model=$MODEL db=$DATASET
```

## License

The software is licensed under the Apache 2.0 license (see [LICENSE](https://github.com/azu-biopharmaceuticals-rd/uq4dd/blob/main/LICENSE)) and is free and provided as-is.

## Contributors
- [Emma Svensson](https://github.com/emmas96)
- [Rosa Friesacher](https://github.com/hannahrosafriesacher)

## Citation

Please cite our work using the following references.

For application in classification,
```bibtex
@inproceedings{friesacher2024towards,
    title={Towards Reliable Uncertainty Estimates for Drug Discovery: A Large-scale Temporal Study of Probability Calibration},
    author={Hannah Rosa Friesacher and Emma Svensson and Adam Arany and Lewis Mervin and Ola Engkvist},
    booktitle={ICML 2024 AI for Science Workshop},
    year={2024},
    url={https://openreview.net/forum?id=5B8tsitI5s}
}
```

For application in regression,
```bibtex
@article{svensson2024enhancing,
    title={Enhancing Uncertainty Quantification in Drug Discovery with Censored Regression Labels},
    author={Svensson, Emma and Friesacher, Hannah Rosa and Winiwarter, Susanne and Mervin, Lewis and Arany, Adam and Engkvist, Ola},
    journal={arXiv preprint arXiv:2409.04313},
    year={2024}
}
```

*Additionally*, the work has been presented as workshop papers at ICANN 2024:

Friesacher, H. R., et al. "Temporal Evaluation of Probability Calibration with Experimental Errors." International Workshop on AI in Drug Discovery. Cham: Springer Nature Switzerland, 2024.

Svensson, E., et al. "Temporal Evaluation of Uncertainty Quantification under Distribution Shift." International Workshop on AI in Drug Discovery. Cham: Springer Nature Switzerland, 2024.

## Funding

The work behind this package has received funding from the European Union’s Horizon 2020
research and innovation programme under the Marie Skłodowska-Curie
Actions, grant agreement “Advanced machine learning for Innovative Drug
Discovery (AIDD)” No 956832”. [Homepage](https://ai-dd.eu/).

![plot](https://github.com/MolecularAI/uq4dd/blob/main/illustrations/aidd.png)

## References
Amini, A., et al. "Deep evidential regression." Advances in Neural Information Processing Systems 33 (2020): 14927-14937.

Arany, A., et al. "SparseChem: Fast and accurate machine learning model for small molecules." arXiv preprint arXiv:2203.04676 (2022).

Blundell, C., et al. "Weight uncertainty in neural network." International Conference on Machine Learning. PMLR, 2015.

Gal, Y., and Ghahramani, Z. "Dropout as a bayesian approximation: Representing model uncertainty in deep learning." International Conference on Machine Learning. PMLR, 2016.

Lakshminarayanan, B., Pritzel, A., and Blundell, C. "Simple and scalable predictive uncertainty estimation using deep ensembles." Advances in Neural Information Processing Systems 30 (2017).

Nix, D. A., and Weigend, A. S. "Estimating the mean and variance of the target probability distribution." Proceedings of 1994 IEEE International Conference on Neural Networks (ICNN'94). Vol. 1. IEEE, 1994.

Platt, J. "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods." Advances in Large Margin Classifiers 10.3 (1999): 61-74.

Rasmussen, M. H., et al. "Uncertain of uncertainties? A comparison of uncertainty quantification metrics for chemical data sets." Journal of Cheminformatics 15.1 (2023): 121.

Vovk, V., and Petej I. "Venn-Abers predictors." Proceedings of the Thirtieth Conference on Uncertainty in Artificial Intelligence. 2014.

## Keywords
uncertainty quantification, probability calibration, deep learning, drug discovery, molecular property prediction

