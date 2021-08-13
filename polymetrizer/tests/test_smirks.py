import pytest

from rdkit import Chem

from polymetrizer import Monomer
from polymetrizer.smirks import BeSmirker, SmirkSet
from .smirks import TRP_SMIRKS


@pytest.fixture()
def rdtrp():
    mon = Monomer.from_smiles("C1=CC=C2C(=C1)C(=CN2)CC(C(=O)[O-])N")
    return mon.to_rdkit()


@pytest.mark.parametrize("label_atom_hydrogen_count, indices", [
    (True, ((0, 1, 2, 3, 4, 5, 6, 7, 8,
             9, 10, 11, 12, 13, 14),)),
    (False, ((0, 1, 2, 3, 4, 5, 18, 6, 7, 8, 20,
              19, 9, 10, 11, 12, 13, 14, 24, 25,
              23, 21, 22, 17, 16, 15),)),
])
@pytest.mark.parametrize("label_atom_element", [True, False])
@pytest.mark.parametrize("label_atom_aromaticity", [True, False])
@pytest.mark.parametrize("label_atom_connectivity", [True, False])
@pytest.mark.parametrize("label_ring_connectivity", [True, False])
@pytest.mark.parametrize("label_ring_atoms", [True, False])
@pytest.mark.parametrize("label_atom_formal_charge", [True, False])
def test_besmirker(rdtrp, label_atom_element,
                   label_atom_aromaticity,
                   label_atom_hydrogen_count,
                   label_atom_connectivity,
                   label_ring_connectivity,
                   label_ring_atoms,
                   label_atom_formal_charge,
                   indices):
    smirker = BeSmirker(label_atom_element=label_atom_element,
                        label_atom_aromaticity=label_atom_aromaticity,
                        label_atom_hydrogen_count=label_atom_hydrogen_count,
                        label_atom_connectivity=label_atom_connectivity,
                        label_ring_connectivity=label_ring_connectivity,
                        label_ring_atoms=label_ring_atoms,
                        label_atom_formal_charge=label_atom_formal_charge)
    output = smirker(rdtrp, label_atom_numbers=[1])
    KEY = (label_atom_element,
           label_atom_aromaticity,
           label_atom_hydrogen_count,
           label_atom_connectivity,
           label_ring_connectivity,
           label_ring_atoms,
           label_atom_formal_charge)
    assert output == TRP_SMIRKS[KEY]

    query = Chem.MolFromSmarts(output)
    assert rdtrp.GetSubstructMatches(query) == indices
