# PolymerGasMembraneML (pgmML)
A machine-learning framework that can learn a an accurate, interpretable model between polymer chemistry and membrane gas permeability. Please refer to our work "Discovery of High-Performance Polymers for Gas-Separation Membranes using Interpretable Machine Learning" for additional details.

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
However, for calculating chemical descriptors and fingerprints, a separate rdkit environment is required.

## General Use
Referring to Figure 1 in our paper, there are 5 steps in our ML training and discovery workflow. Step 1, curating a dataset of SMILES strings and permeabities, has been accomplished and the results are contained in the 'datasets' folder. Step 2, generating chemical representations via RDKit, is an optional step and is only necessary for the screnning datasets (Datasets B and C), as the training set has already been converted. Step 3 is training ML models, 
