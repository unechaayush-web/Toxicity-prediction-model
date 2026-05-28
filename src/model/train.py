print('Toxicity model training')
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,f1_score,precision_score,recall_score
from imblearn.over_sampling import SMOTE

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import Descriptors,rdMolDescriptors

# Load dataset
data = pd.read_csv("Data/tox21.csv")

df = data.drop("mol_id", axis=1)
df = df.dropna()

print("Dataset Loaded")
print(df.head())


# Fingerprint generator
generator = GetMorganGenerator(radius=2, fpSize=2048)

def generate_features(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return np.zeros(2055)

    # fingerprint
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((2048,))
    DataStructs.ConvertToNumpyArray(fp, arr)

    # descriptors
    desc = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol)
    ]

    return np.concatenate([arr, desc])

# Features
X = np.array(df["smiles"].apply(generate_features).tolist())

# Labels
y = df.drop("smiles", axis=1)


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

Scaler = StandardScaler()
X_train_scale = Scaler.fit_transform(X_train)
X_test_scale = Scaler.transform(X_test)

# Model
base_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced_subsample",
    random_state=42
)

model = MultiOutputClassifier(base_model)

print("Training model...")

model.fit(X_train_scale, y_train)

print("Model training done")


# Cross Validation
scores = cross_val_score(model, X_train_scale, y_train, cv=5)

print("Cross Validation Scores:", scores)
print("Mean Accuracy:", scores.mean())


# Prediction Probabilities
print("\nChecking prediction probabilities...")

probs = model.predict_proba(X_test_scale)


# Convert probability → prediction
threshold = 0.02   # adjust based on your probability range

y_pred = []

for i, col in enumerate(y.columns):

    print(f"\nToxicity endpoint: {col}")

    toxic_prob = probs[i][:, 1]

    print("First 10 toxic probabilities:")
    print(toxic_prob[:10])

    pred = (toxic_prob > threshold).astype(int)

    y_pred.append(pred)

y_pred = np.array(y_pred).T


# Metrics
print("\nClassification Reports:")

for i, col in enumerate(y.columns):

    print("\nEndpoint:", col)

    print(
        classification_report(
            y_test[col],
            y_pred[:, i]
        )
    )

# Label Distribution
print("\nLabel Distribution:")

for col in y.columns:
    print(col)
    print(y[col].value_counts())
    print()

# Save model
joblib.dump(model, "toxicity_model.pkl")

print("Model saved as toxicity_model.pkl")
