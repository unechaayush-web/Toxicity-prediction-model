print('Toxicity prdiction')
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdMolDescriptors
import numpy as np
import joblib


# Load model and scaler
model = joblib.load("toxicity_model.pkl")


# Molecular descriptors
def Smiles_info(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol),
        "RotB": Descriptors.NumRotatableBonds(mol),
        "Rings": rdMolDescriptors.CalcNumRings(mol),
    }


# Morgan fingerprint
def generate_fingerprint(smiles, radius=2, n_bits=2048):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return np.zeros(n_bits)

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=n_bits
    )

    arr = np.zeros((n_bits,))
    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr


# Toxicity prediction
def Toxicity_prediction(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return "Invalid SMILES"

    info = Smiles_info(smiles)

    descriptors = np.array(list(info.values())).reshape(1, -1)

    fingerprint = generate_fingerprint(smiles).reshape(1, -1)

    # combine features
    features = np.concatenate((fingerprint, descriptors), axis=1)

    # prediction
    prediction = model.predict(features)

    probabilities = model.predict_proba(features)

    prediction_list = prediction[0].tolist()

    labels = [
      'Nuclear Receptor- Androgen Receptor',                               
      'Nuclear Receptor-Androgen Receptor Ligand Binding Domain',          
      'Nuclear Receptor-Aryl Hydrocarbon Receptor',                         
      'Nuclear Receptor-Aromatase Enzyme',                                 
      'Nuclear Receptor-Estrogen Receptor',                                
      'Nuclear Receptor-Estrogen Receptor Ligand Binding Domain',         
      'Nuclear Receptor-Peroxisome Proliferator Activated Receptor Gamma', 
      'Stress Response-Antioxidant Response Element',                      
      'Stress Response-ATPase Family AAA Domain Containing 5',             
      'Stress Response-Heat Shock Element',                                
      'Stress Response-Mitochondrial Membrane Potential',                  
      'Stress Response- Tumor Protein p53',                                 
    ]

    print("\nToxicity Prediction Probabilities:")

    for label, prob in zip(labels, probabilities):

        toxic_prob = prob[0][1]

        print(f"{label}: {toxic_prob:.3f}")

    return prediction_list


# CLI Interface
while True:

    print("\nHello I am Ultron your toxicity predictor model")

    smiles = input("Enter SMILES (or type exit): ")

    if smiles.lower() == "exit":
        print('Thanks for using')
        break

    info = Smiles_info(smiles)

    if info is None:
        print("Invalid SMILES")
        continue

    print("\nMolecule Info:")
    print(info)

    Toxicity_prediction(smiles)
