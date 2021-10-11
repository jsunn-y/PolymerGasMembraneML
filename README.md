# PolymerGasMembraneML (pgmML)
A machine-learning framework that can learn an accurate, interpretable model between polymer chemistry and membrane gas permeability. Please refer to our work "Discovery of High-Performance Polymers for Gas-Separation Membranes using Interpretable Machine Learning" for additional details.

## Installation
To download, clone this repository
```
git clone https://github.com/jsunn-y/PolymerGasMembraneML
```
To run any code does not require chemical feature calculation via rdkit, the anaconda environment can be downloaded from `pgmML.yml`. To build this environment, run
```
cd ./PolymerGasMembraneML
conda env create -f pgmML.yml
```
However, for calculating chemical descriptors and fingerprints, a separate rdkit environment is required. Installation instructions can be found [here](https://www.rdkit.org/docs/Install.html).

## Datasets
In addition to the training dataset, Dataset A, we use 3 screening datasets in this work: Dataset B, C, and D. Due to large size of Datasets B and C, we do not include any calculated features in our hub, and for Dataset C, we only include the first 1 million SMILES strings. 
However all the datasets used in this work, including smiles and calculated fingerprints, can be downloaded [here](https://drive.google.com/file/d/1NPh3Hx3nHakUH4bgp24Ie1KCEAvZnCr4/view?usp=sharing).

## General Use
Referring to Figure 1 in our paper, there are 5 steps in our ML training and discovery workflow. Step 1, curating a dataset of SMILES strings and permeabities, has been accomplished and the results are contained in the `/datasets` folder. Step 2, generating chemical representations via RDKit, is an optional step as all fingerprints have been generated and uploaded [at the same link above](https://drive.google.com/file/d/1NPh3Hx3nHakUH4bgp24Ie1KCEAvZnCr4/view?usp=sharing). The chemical features of the training set have already been computed and uploaded to the `/datasets` folder. 

We believe that this repository will be most useful for Step3-4, training, interpretation, and screening.
We recommend training our top-performing ML model, a DNN ensemble trained using Morgan fingerprints as inputs with permeabilities imputed using Bayesian Linear Regression in step 3:
```
python step3_train.py --features 'fing' --imputation 'BLR' --model 'DNN'
```
Alternatively, one can also train on descriptors, use extremely randomized trees for imputation, and train a random forest. We also include several pretrained models in `/pretrained_models` to replicate similar results to those demonstrated in our paper.

Using the saved models, it is possible to obtain the model's SHAP values in step 3.5. For example, run:
```
python step3.5_SHAP.py --modelname 'DNN_BLR_fing'
```
As part of the discovery framework, we also screen large datasets of polymers to predict their permeabilities and identify promising candidates. To screen Dataset B on the model that we trained and analyzed above, run:
```
python step4_screen.py --modelname 'DNN_BLR_fing' --dataset 'datasetBX_fing.csv'
```
## Visualization of Results
The performance of a trained model can be verified via:
```
python plot_modelR2.py --modelname 'DNN_BLR_fing'
```
Similarly, the shap values for a trained model can be visualized via:
```
python plot_SHAP.py --modelname 'DNN_BLR_fing'
```
We also offer a template `plot_robeson.py` to visualize permeabilities for O_{2}/N_{2} and CO_2/CH_4 separations in context.

