# gasseparationML

Code for benchmarking ML models for synthesis-property relationships of polymer gas separation membranes using supervised and unsupervised learning. Please refer to the manuscript for additional details.

### Relevant Datasets:
**dataset1_imputed_all.csv**: training set containing SMILES string inputs, raw premeabilities, and imputer permeabilities among other information

**dataset2.csv**: screening dataset with SMILES strings

**dataset3.csv**: screening dataset with SMILES strings


**dataset1X_descriptors.csv**: calculated descriptors for dataset1

**dataset2X_descriptors.csv**: calculated descriptors for dataset2

**dataset1X_fingerprints.csv**: calculated fingerprints for dataset1

**dataset2X_fingerprints.csv**: calculated fingerprints for dataset2

### Code:
**supervisedML.ipynb**: jupyter notebook to train all of the supervised ML models used in the work

**FMGAN.ipynb**: jupyter notebook to train the unsupervised FMGAN

