import pytest
from hypothesis import given, strategies as st
from openff.toolkit.topology import Molecule as OFFMolecule
import numpy as np

import polymetrizer as pet
from polymetrizer.oligomer import (Oligomer, Monomer,
                                   AtomWrapper, HYDROGEN,
                                   create_hydrogen_caps)


from .data import PEGMA_R_SMILES,  BMA_R_SMILES

CAPLIST = [(1, HYDROGEN), (2, HYDROGEN), (3, HYDROGEN)]
SUBSTITUENTS = {i: CAPLIST for i in range(1, 10)}

@given(r_numbers=st.sets(st.integers()))
def test_create_hydrogen_caps(r_numbers):
    r_group_numbers = list(r_numbers)
    caps = create_hydrogen_caps(r_group_numbers)
    assert list(caps) == r_group_numbers
    assert all(len(x) == 1 for x in caps.values())
    assert all(x[0][0] == 1 for x in caps.values())

class TestMonomer:
    @pytest.mark.parametrize(
        "smiles, r_group_indices, central_indices",
        [
            ("([1*:1])[H]", {1: 0}, [1]),
            ("([R1])[H]", {1: 0}, [1]),
            ("([*:3])[H]", {3: 0}, [1]),
            ("[R3][H]", {3: 0}, [1]),
            ("C([1*:1])([*:2])=N([*:3])", {1: 1, 2: 2, 3: 4}, [0, 3]),
            ("C([R1])([R2])=N([R3])", {1: 1, 2: 2, 3: 4}, [0, 3]),
        ]
    )
    def test_create_monomer_from_smiles(self, smiles, r_group_indices, central_indices):
        monomer = Monomer(smiles)
        assert monomer.r_group_indices == r_group_indices
        assert monomer.central_atom_indices == central_indices
        assert sorted(monomer.atom_oligomer_map) == central_indices
        for index, wrapper in monomer.atom_oligomer_map.items():
            assert wrapper.index == index
            assert wrapper.monomer is monomer
        
        another = Monomer(smiles)
        # assert monomer == another
        # assert hash(monomer) != hash(another)
    

@pytest.fixture
def pegma():
    monomer = Monomer(PEGMA_R_SMILES)
    assert monomer.offmol.n_atoms == 45
    assert set(monomer.r_group_indices.keys()) == {1, 2}
    assert len(monomer.atom_oligomer_map) == 43
    return monomer

@pytest.fixture
def bma():
    return Monomer(BMA_R_SMILES)

@pytest.fixture
def pegma_pegma(pegma):
    return pegma.attach_substituents({1: (2, pegma)})

PEGMA_PEGMA_CENTRAL_INDICES = [44, 45, 46, 47, 49] + list(range(50, 88))

class TestOligomer:

    @pytest.mark.parametrize("smiles, r_group_indices", [
        ("([1*:1])[H]", {1: 0}),
        ("([*:3])[H]", {3: 0}),
        ("C([1*:1])([*:2])=N([*:3])", {1: 1, 2: 2, 3: 4})
    ])
    def test_create_oligomer_without_maps(self, smiles, r_group_indices):
        offmol = OFFMolecule.from_smiles(smiles, allow_undefined_stereo=True)
        oligomer = Oligomer(offmol)
        assert oligomer.r_group_indices == r_group_indices

    @pytest.mark.parametrize("smiles, index", [
        ("([1*])[H]", 0),
        ("([*])[H]", 0),
        ("C([1*:1])([*:2])=N([*])", 4)
    ])
    def test_create_oligomer_with_unlabelled(self, smiles, index):
        offmol = OFFMolecule.from_smiles(smiles, allow_undefined_stereo=True)
        err = f"must be labelled. The atom at index {index}"
        with pytest.raises(ValueError, match=err):
            oligomer = Oligomer(offmol)
    
    def test_get_applicable_caps_no_ignore(self, pegma):
        caps = pegma.get_applicable_caps(SUBSTITUENTS)
        assert len(caps) == 9
        assert all(set(cap.keys()) == {1, 2} for cap in caps)
    
    def test_get_applicable_caps_ignore_r(self, pegma):
        caps = pegma.get_applicable_caps(SUBSTITUENTS, ignore_r=1)
        assert len(caps) == 3
        assert all(set(cap.keys()) == {2} for cap in caps)
    
    def test_get_cap_combinations(self, pegma):
        combs = pegma.get_cap_combinations(SUBSTITUENTS)
        assert set(combs.keys()) == {1, 2}
        assert all(set(cap.keys()) == {2} for cap in combs[1])
        assert all(set(cap.keys()) == {1} for cap in combs[2])

    def test_attach_substituents_h(self, pegma):
        new = pegma.attach_substituents({1: (1, HYDROGEN)})
        assert new.offmol.n_atoms == 45
        assert set(new.r_group_indices.keys()) == {2}
        expected_central = [0, 1, 2, 4, 6] + list(range(7, 45))
        assert new.central_atom_indices == expected_central
        assert new.offmol.atoms[3].atomic_number == 1
        assert new.offmol.atoms[5].atomic_number == 0
        assert new.r_group_indices[2] == 5
        assert new.atom_oligomer_map[3].monomer is HYDROGEN
        for i in expected_central:
            assert new.atom_oligomer_map[i].monomer is pegma

    def test_attach_substitutents_pegma(self, pegma_pegma):
        assert pegma_pegma.offmol.n_atoms == 88
        assert set(pegma_pegma.r_group_indices.keys()) == {1, 2}
        assert pegma_pegma.central_atom_indices == PEGMA_PEGMA_CENTRAL_INDICES
        assert pegma_pegma.offmol.atoms[48].atomic_number == 0
        assert pegma_pegma.r_group_indices[2] == 48
        assert pegma_pegma.offmol.atoms[0].atomic_number == 0
        assert pegma_pegma.r_group_indices[1] == 0

    @pytest.mark.parametrize("n_neighbors, indices", [
        (0, PEGMA_PEGMA_CENTRAL_INDICES),
        (1, PEGMA_PEGMA_CENTRAL_INDICES + [4]),
        (2, PEGMA_PEGMA_CENTRAL_INDICES + [1, 4, 5, 40]),
        (3, PEGMA_PEGMA_CENTRAL_INDICES + [1, 2, 3, 4, 5, 6, 7, 40, 41, 42, 43])
    ])
    def test_get_central_and_neighbor_indices(self, pegma_pegma, n_neighbors, indices):
        assert pegma_pegma.get_central_and_neighbor_indices(n_neighbors) == sorted(indices)

    def test_generate_substituted_caps(self, pegma):
        hydrogens = create_hydrogen_caps(pegma.r_group_indices)
        caps = pegma.generate_substituted_caps(hydrogens)
        assert len(caps) == 2