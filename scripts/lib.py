# -*- coding: utf-8 -*-

"""Library of common functions for scripting."""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def smiles_to_mol(smiles: str) -> Chem.Mol:
    """Convert a SMILES string to an RDKit molecule.

    :param smiles: The SMILES string to convert.
    :type smiles: str
    :return: The RDKit molecule.
    :rtype: Chem.Mol
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol


def mol_to_fingerprint(mol: Chem.Mol, radius: int = 2, num_bits: int = 1024) -> np.ndarray:
    """Convert an RDKit molecule to a Morgan fingerprint.

    :param mol: The RDKit molecule to convert.
    :type mol: Chem.Mol
    :return: The Morgan fingerprint.
    :rtype: np.ndarray
    """
    fp_arr = np.zeros((0,), dtype=np.int8)
    fp_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    DataStructs.ConvertToNumpyArray(fp_vec, fp_arr)
    return fp_arr
