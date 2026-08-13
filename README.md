# Stochastic Modeling of Composite Interfaces

This repository contains the numerical implementation, post-processing scripts, and normalized experimental datasets supporting the research paper:

> **"Stochastic Modeling of Composite Interfaces: Sensitivity to Spatial Correlation and Bayesian Identification from Standard Fracture Tests"**  
> Published in *Integrating Materials and Manufacturing Innovation* (IMMI), Topical Collection *"Uncertainty Quantification for Decision Support in ICME"*.

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
│   ├── DCB_normalized_curves.csv    # Normalized experimental data for Mode I (DCB)
│   └── ENF_normalized_curves.csv    # Normalized experimental data for Mode II (ENF)
├── src/
│   ├── random_fields.py             # Stationary Gaussian random field generator
│   └── abc_knn.py                   # ABC-kNN Bayesian inference framework
├── scripts/
│   ├── reproduce_sensitivity.py     # Sensitivity analysis scripts
│   ├── reproduce_bayes_dcb.py       # ABC-kNN parameter identification for Mode I
│   └── reproduce_figures.py         # Plotting scripts to regenerate paper figures
├── requirements.txt                 # Python dependencies
├── LICENSE                          # License file
└── README.md                        # Project documentation
