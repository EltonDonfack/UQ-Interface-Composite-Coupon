# Stochastic Modeling of Composite Interfaces

This repository contains the numerical implementation and post-processing scripts supporting the research paper:

> **"Stochastic Modeling of Composite Interfaces: Sensitivity to Spatial Correlation and Bayesian Identification from Standard Fracture Tests"**  
> Published in *Integrating Materials and Manufacturing Innovation* (IMMI), Topical Collection *"Uncertainty Quantification for Decision Support in ICME"*.  
> **Preprint:** [arXiv:2606.13019](https://arxiv.org/abs/2606.13019)
> 

## Authors
* **Elton Donfack-Siewe** (Airbus Operations SAS / ONERA)
* **Sylvain Dubreuil** (ONERA)
* **Christian Fagiano** (ONERA)
* **Jérôme Morio** (ONERA)
* **Jean-Philippe Navarro** (Airbus Operations SAS)

---

## 📁 Repository Structure

```text
├── src/
│   ├── random_fields.py                 # Stationary Gaussian random field generator
│   └── abc_knn.py                       # ABC-kNN Bayesian inference framework
├── scripts/
│   ├── plot_marginal_posteriors.py      # Plots 1D marginal posterior distributions
│   └── plot_posterior_correlation.py    # Plots 5x5 posterior cross-correlation matrix
├── requirements.txt                     # Python dependencies
├── LICENSE                              # MIT License file
└── README.md                            # Project documentation
