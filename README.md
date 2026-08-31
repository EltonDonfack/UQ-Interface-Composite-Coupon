# Stochastic Modeling of Composite Interfaces

This repository contains the normalized experimental test data (DCB and ENF), numerical implementation, and post-processing scripts supporting the research paper:

> **"Stochastic Modeling of Composite Interfaces: Sensitivity to Spatial Correlation and Bayesian Identification from Standard Fracture Tests"**  
> Published in *Integrating Materials and Manufacturing Innovation* (IMMI), Topical Collection *"Uncertainty Quantification for Decision Support in ICME"*.  
> **Preprint:** [arXiv:2606.13019](https://arxiv.org/abs/2606.13019)

## Authors
* **Elton Donfack-Siewe** (Airbus Operations SAS / ONERA)
* **Sylvain Dubreuil** (ONERA)
* **Christian Fagiano** (ONERA)
* **Jérôme Morio** (ONERA)
* **Jean-Philippe Navarro** (Airbus Operations SAS)

---

## 📁 Repository Structure

```text
├── data/
│   ├── DCB/
│   │   ├── Normalized_dcb_test_data_1.csv
│   │   ├── Normalized_dcb_test_data_2.csv
│   │   ├── Normalized_dcb_test_data_3.csv
│   │   ├── Normalized_dcb_test_data_4.csv
│   │   └── Normalized_dcb_test_data_5.csv
│   └── ENF/
│       ├── Normalized_enf_test_data_1.csv
│       ├── Normalized_enf_test_data_2.csv
│       ├── Normalized_enf_test_data_3.csv
│       ├── Normalized_enf_test_data_4.csv
│       └── Normalized_enf_test_data_5.csv
├── scripts/
│   ├── plot_marginal_posteriors.py     # Plots 1D marginal posterior distributions
│   └── plot_posterior_correlation.py    # Plots 5x5 posterior cross-correlation matrix
├── src/
│   ├── random_fields.py               # Stationary Gaussian random field generator
│   └── abc_knn.py                     # ABC-kNN Bayesian inference framework
├── LICENSE                            # MIT License file
├── README.md                          # Project documentation
└── requirements.txt                   # Python dependencies
