# PolymerGasMembraneML (pgmML)
A machine-learning framework that can learn an accurate, interpretable model between polymer chemistry and membrane gas permeability. Please refer to our work "Discovery of High-Performance Polymers for Gas-Separation Membranes using Interpretable Machine Learning" for additional details.

## Installation
To download, clone this repository
```
git clone https://github.com/jsunn-y/PolymerGasMembraneML
```
To run most of the code, the anaconda environment can be downloaded from `pgmML.yml`. To build this environment, run
```
cd ./PolymerGasMembraneML
conda env create -f pgmML.yml
```
However, for calculating chemical descriptors and fingerprints, a separate rdkit environment is required. Installation instructions can be found [here](https://www.rdkit.org/docs/Install.html).

## Datasets
In addition to the training dataset, Dataset A, we use 3 screening datasets in this work: Dataset B, C, and D. Due to large size of Datasets B and C, we do not include any calculated features in our hub, and for Dataset C, we only include the first 1 million SMILES strings. 
However all the datasets used in this work, including smiles and calculated fingerprints, can be downloaded [here](https://drive.google.com/file/d/1NPh3Hx3nHakUH4bgp24Ie1KCEAvZnCr4/view?usp=sharing).

## General Use
Referring to Figure 1 in our paper, there are 5 steps in our ML training and discovery workflow. Step 1, curating a dataset of SMILES strings and permeabities, has been accomplished and the results are contained in the 'datasets' folder. Step 2, generating chemical representations via RDKit, is an optional step as all fingerprints have been generated and uploaded [at the same link above](https://drive.google.com/file/d/1NPh3Hx3nHakUH4bgp24Ie1KCEAvZnCr4/view?usp=sharing). The chemical features of the training set have already been computed and uploaded to the 'datasets' folder. Step 3 is training ML models, 
