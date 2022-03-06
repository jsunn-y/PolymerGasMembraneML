import numpy as np
import pandas as pd
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import os

'''
Script to choose the pertinent molecular descriptors or fingerprints for Dataset A (training) and calculate these chemical features for another dataset base on SMILES strings.
Chemical features are already calculated for Dataset A, so training can begin without this step.
Representations are saved as .csv files with columns as features and rows as samples.
'''

def calculate_representations(args):
    os.chdir(os.getcwd() + '/datasets/')
    DatasetA_Smiles_P = pd.read_csv("datasetA_imputed_all.csv")
    DatasetA_grouped = DatasetA_Smiles_P.groupby('Smiles').mean().reset_index()
    Dataset = pd.read_csv(args.dataset + '.csv')
    
    if args.features == 'desc':
        #get the descriptors from SMILES for Dataset A
        X_descriptors = np.zeros((DatasetA_grouped.shape[0], 208))
        X_descriptors = pd.DataFrame(X_descriptors)
        for i in range(DatasetA_grouped.shape[0]):
            mol = Chem.MolFromSmiles(DatasetA_grouped.loc[i, 'Smiles'])
            X_descriptors.iloc[i,:] = [f[1](mol) for f in Descriptors.descList]

        #drop the columsn with all zeros or with any missing values
        X_descriptors = X_descriptors.dropna(axis='columns')
        X_descriptors = X_descriptors.loc[:, (X_descriptors != 0).any(axis=0)]
        selected_keys = X_descriptors.columns

        #get the descriptors from SMILES for the new dataset
        datasetX_descriptors = np.zeros((Dataset.shape[0], 208))
        datasetX_descriptors = pd.DataFrame(datasetX_descriptors)
        for i, smiles in enumerate(Dataset['Smiles']):
            mol = Chem.MolFromSmiles(smiles)
            datasetX_descriptors.iloc[i,:] = [f[1](mol) for f in Descriptors.descList]
        datasetX_descriptors = datasetX_descriptors[selected_keys]
        filename = args.dataset + '_X_desc.csv'
        datasetX_descriptors.to_csv(filename, index=False)
        print('Features saved to /datasets/'+ filename)

    if args.features == 'fing':
        #dataset-A fingerprint
        molecules = DatasetA_grouped.Smiles.apply(Chem.MolFromSmiles)
        fp = molecules.apply(lambda m: AllChem.GetMorganFingerprint(m, radius=3))
        fp_n = fp.apply(lambda m: m.GetNonzeroElements())

        # using substructures in dataset-A to construct a dictionary
        HashCode = []
        for i in fp_n:
            for j in i.keys():
                HashCode.append(j)
                
        unique_set = set(HashCode)
        unique_list = list(unique_set)

        Corr_df = pd.DataFrame(unique_list).reset_index()

        #construct dataset-A input
        MY_finger = []
        for polymer in fp_n:
            my_finger = [0] * len(unique_list)
            for key in polymer.keys():
                index = Corr_df[Corr_df[0] == key]['index'].values[0]
                my_finger[index] = polymer[key]
            MY_finger.append(my_finger)
            
        MY_finger_dataset_A = pd.DataFrame(MY_finger)

        # filter input into the most popular 114 substructures
        Zero_Sum = (MY_finger_dataset_A == 0).astype(int).sum()
        NumberOfZero = 325 #adjust this number based on the tolerance of how many substructures
        print(len(Zero_Sum[Zero_Sum < NumberOfZero]))
        X_fingerprints = MY_finger_dataset_A[Zero_Sum[Zero_Sum < NumberOfZero].index]
        new_length = X_fingerprints.shape[1]

        selected_keys = X_fingerprints.columns
        selected_Corr_df = Corr_df.iloc[selected_keys,-1]
        selected_keys = Corr_df.iloc[selected_keys,-1].to_numpy()

        #get fingerprints for the new dataset
        molecules = Dataset.Smiles.apply(Chem.MolFromSmiles)
        fp = molecules.apply(lambda m: AllChem.GetMorganFingerprint(m, radius=3))
        fp_n = fp.apply(lambda m: m.GetNonzeroElements())

        #construct new dataset input
        MY_finger = []
        for polymer in fp_n:
            my_finger = [0] * new_length
            for key in polymer.keys():
                if key in selected_keys:
                    index = np.where(selected_keys == key)[0][0]
                    my_finger[index] = polymer[key]
            MY_finger.append(my_finger)
            
        MY_finger_dataset = pd.DataFrame(MY_finger)  
        MY_finger_dataset.columns = selected_Corr_df.index
        filename = args.dataset + '_X_fing.csv'
        MY_finger_dataset.to_csv(filename, index=False)
        print('Features saved to /datasets/'+ filename)

#for parser arguments
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, required = True, 
    	help='Choose either "desc" for chemical descriptors or "fing" for fingerprints as the input features')
    parser.add_argument('--dataset', type=str, required = True, 
    	help='Specify the filename without csv (datasetB.csv or datasetC.csv) within the dataset folder to compute the features of. This files should have a single column of SMILES Strings labeled with "Smiles"')        

    parsed_args = parser.parse_args()

    calculate_representations(parsed_args)