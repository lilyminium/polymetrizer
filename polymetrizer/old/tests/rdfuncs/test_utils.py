import pytest


import polymetrizer as pet


@pytest.mark.parametrize("smiles, n_atoms", [
    ("[CH3][CH2][OH]", 3),  # no hs
    ("[H][O][C]([H])([H])[C]([H])([H])[H]", 9),  # all hs
    ("[C:1]([C:2]([O:3][H:9])([H:7])[H:8])([H:4])([H:5])[H:6]", 9),  # mapping
    ("[4H][1C]([5H])([6H])[2C]([7H])([8H])[3O][9H]", 9),  # isotopes
    ("[1C:2]([2C:3]([3O:4][9H:10])([7H:8])[8H:9])([4H:5])([5H:6])[6H:7]", 9),  # both
])
def test_rdmol_smiles_roundtrip(smiles, n_atoms):
    rdmol = pet.rdfuncs.utils.mol_from_smiles(smiles)
    assert rdmol.GetNumAtoms() == n_atoms
    assert pet.rdfuncs.utils.mol_to_smiles(rdmol) == smiles


