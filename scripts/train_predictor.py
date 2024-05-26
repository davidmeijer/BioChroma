# -*- coding: utf-8 -*-

"""Script to train a predictor for maximum absorbance wavelength."""

import joblib
import typing as ty
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from lib import (
    smiles_to_mol, 
    mol_to_fingerprint
)


def predict(predictor: ty.Any, name: str, smiles: str, true: float) -> float:
    """Predict the maximum absorbance wavelength of a compound.

    :param predictor: The trained predictor.
    :type predictor: ty.Any
    :param name: The name of the compound.
    :type name: str
    :param smiles: The SMILES string of the compound.
    :type smiles: str
    :param true: The true maximum absorbance wavelength.
    :type true: float
    """
    mol = smiles_to_mol(smiles)
    fingerprint = mol_to_fingerprint(mol)
    wavelength = predictor.predict([fingerprint])[0]
    msg = f"Predicted wavelength of {name}: {wavelength:.2f} nm (true: ~{true:.2f} nm)"
    print(msg)


def main() -> None: 
    """Main function for the script."""

    # Parse data.
    path = "./temp/joonyoung.csv"

    compound_fingerprints = []
    compound_wavelengths = []

    with open(path) as file_open:
        file_open.readline()

        for line in file_open:
            smiles_id, smiles, solvent, wavelength, *_ = line.strip().split(",")
            
            # if solvent == "ClCCl":
            mol = smiles_to_mol(smiles)
            fingerprint = mol_to_fingerprint(mol)

            compound_fingerprints.append(fingerprint)
            compound_wavelengths.append(float(wavelength))

    compound_fingerprints = np.array(compound_fingerprints)
    compound_wavelengths = np.array(compound_wavelengths)
    print(compound_fingerprints.shape, compound_wavelengths.shape)

    # Check for NaNs.
    nan_indices = np.isnan(compound_wavelengths)
    print(f"Found {np.sum(nan_indices)} NaNs in wavelength data.")

    compound_fingerprints = compound_fingerprints[~nan_indices]
    compound_wavelengths = compound_wavelengths[~nan_indices]
    print(compound_fingerprints.shape, compound_wavelengths.shape)

    # Show distribution of wavelengths.
    counter = Counter(compound_wavelengths)
    x = list(counter.keys())
    y = list(counter.values())

    plt.bar(x, y, label="Wavelengths", color="gray")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Count")
    plt.title("Distribution of Wavelengths")

    plt.axvline(x=380, color="k", linestyle="--", label="Visible spectrum")
    plt.axvline(x=700, color="k", linestyle="--", label="Visible spectrum") 
    plt.legend(["Visible spectrum"])

    # plt.show()
    plt.savefig("./temp/wavelengths.png")

    # Partition data.
    num_compounds = compound_fingerprints.shape[0]
    indices = np.random.permutation(num_compounds)
    num_train = int(0.8 * num_compounds)

    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    train_fingerprints = compound_fingerprints[train_indices]
    train_wavelengths = compound_wavelengths[train_indices]

    test_fingerprints = compound_fingerprints[test_indices]
    test_wavelengths = compound_wavelengths[test_indices]

    print(train_fingerprints.shape, train_wavelengths.shape)
    print(test_fingerprints.shape, test_wavelengths.shape)

    # Train predictor.
    predictor = RandomForestRegressor(n_estimators=1000)
    predictor.fit(train_fingerprints, train_wavelengths)

    # Evaluate predictor.
    train_predictions = predictor.predict(train_fingerprints)
    test_predictions = predictor.predict(test_fingerprints)

    train_errors = np.abs(train_predictions - train_wavelengths)
    test_errors = np.abs(test_predictions - test_wavelengths)

    train_mae = np.mean(train_errors)
    test_mae = np.mean(test_errors)

    print(f"Train MAE: {train_mae:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    
    # Save model.
    joblib.dump(predictor, "./temp/predictor.joblib")

    # Predict wavelength for a new compound.
    name = "cholorophyll"
    smiles = "CCC1=C(C2=NC1=CC3=C(C4=C(C(C(=C4[N-]3)C5=NC(=CC6=NC(=C2)C(=C6C)C=C)C(C5CCC(=O)OCC=C(C)CCCC(C)CCCC(C)CCCC(C)C)C)C(=O)OC)[O-])C)C"
    true = 400.0  # Green
    predict(predictor, name, smiles, true)

    name = "indigo"
    smiles = "C1=CC=C2C(=C1)C(=C(N2)C3=NC4=CC=CC=C4C3=O)O"
    true = 440.0  # Dark blue
    predict(predictor, name, smiles, true)

    name = "beta-carotene"
    smiles = "CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC2=C(CCCC2(C)C)C)C)C"
    true = 450.0  # Red / orange
    predict(predictor, name, smiles, true)

    exit(0)

if __name__ == "__main__":
    main()
