import pytest
from hypothesis import given, strategies as st
from openff.toolkit.topology import Molecule as OFFMolecule
import numpy as np

import polymetrizer as pet
from polymetrizer.oligomer import (Oligomer, Monomer,
                                   AtomWrapper, HYDROGEN,
                                   create_hydrogen_caps)


from .data import PEGMA_R_SMILES,  BMA_R_SMILES, SPLIT_MOL_R_SMILES, SPLIT_MOL_R_LINKAGES, FULL_MOL_SMARTS

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
def cap_options(pegma, bma):
    return {1: [(2, bma), (2, pegma), (1, HYDROGEN)],
            2: [(1, bma)]}

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

    def create_oligomer_from_monomer(self, bma):
        oligomer = Oligomer(bma)
        for atom in oligomer.atom_oligomer_map.values():
            assert atom.monomer is bma

    def test_sort_oligomers(self, bma, pegma):
        assert pegma == pegma
        assert bma < pegma

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
        assert len(combs[1]) == 3
        assert all(set(cap.keys()) == {1} for cap in combs[2])
        assert len(combs[2]) == 3

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

    def test_generate_substituted_caps_h(self, pegma):
        hydrogens = create_hydrogen_caps(pegma.r_group_indices)
        caps = pegma.generate_substituted_caps(hydrogens)
        assert len(caps) == 2
        assert len(caps[1]) == 1
        assert len(caps[2]) == 1
        first = caps[1][0]
        assert first.offmol.n_atoms == 45
        assert first.offmol.atoms[5].atomic_number == 0
        second = caps[2][0]
        assert first.offmol.n_atoms == 45
        assert first.offmol.atoms[5].atomic_number == 0

    def test_generate_substituted_caps(self, pegma, cap_options):
        caps = pegma.generate_substituted_caps(cap_options)
        assert len(caps) == 2
        assert len(caps[1]) == 1
        assert len(caps[2]) == 3

    @pytest.fixture
    def substituted(self, pegma, cap_options):
        return pegma.generate_substituted(cap_options)

    @pytest.mark.parametrize("index, n_atoms, r_group_indices, bma_indices", [
        (0,  93, {1: 41, 2: 0}, [i for i in range(25, 37) if i != 29] + list(range(75, 93)) + [40, 49, 50, 51, 53, 54, 55, 59, 63, 64, 65, 69, 70, 71]),
        (1, 112, {1: 41, 2: 0}, list(range(25, 112))),
        (2,  69,        {2: 0}, [i for i in range(25, 69) if i != 31]),
    ])
    def test_generate_substituted(self, pegma, substituted, index, n_atoms, r_group_indices, bma_indices):
        assert len(substituted) == 3
        product = substituted[index]
        assert product.offmol.n_atoms == n_atoms
        assert product.r_group_indices == r_group_indices

        for index, atom in sorted(product.atom_oligomer_map.items()):
            if index in bma_indices:
                assert atom.monomer is pegma
            else:
                assert atom.monomer is not pegma

    @pytest.mark.parametrize("pegma_atom_indices, handler_name, contains", [
        ([1], None, True),
        ([1, 2], None, True),
        ([2, 1, 3], None, True),
        ([2, 3, 1], None, False),
        ([4, 6, 7, 8], None, False),
        ([4, 6, 7, 8], "ImproperTorsions", True),
        ([6, 8, 9, 10], None, True),
        ([6, 8, 9, 10], "ImproperTorsions", False),
    ])
    def test_contains_monomer_atoms(self, pegma, pegma_atom_indices, handler_name, contains):
        atoms = [pegma.atom_oligomer_map[i] for i in pegma_atom_indices]
        result, indices = pegma.contains_monomer_atoms(atoms, handler_name, return_indices=True)
        assert result == contains
        if contains:
            assert indices == [tuple(pegma_atom_indices)]

    @pytest.fixture
    def split_monomers(self):
        return [Monomer(smiles) for smiles in SPLIT_MOL_R_SMILES]

    @pytest.mark.parametrize("root_index", [0, 1, 2, 3, 4])
    def test_build_all_combinations(self, root_index, split_monomers):
        root = split_monomers[root_index]
        caps = {
            1: [(2, split_monomers[1])],
            2: [(1, split_monomers[0])],
            3: [(4, split_monomers[2])],
            4: [(3, split_monomers[1])],
            5: [(6, split_monomers[3])],
            6: [(5, split_monomers[2])],
            7: [(8, split_monomers[4])],
            8: [(7, split_monomers[3])],
        }

        final = root.build_all_combinations(caps)
        assert final.offmol.n_atoms == 94
        assert final.offmol.chemical_environment_matches(FULL_MOL_SMARTS)
    
    def test_select_relevant_parameters(self, pegma):
        handler_kwargs = {
            "bonds": {(1, 2): 3,
                      (1, 3): 4,
                      (2, 3): 5,
                      (2, 4): 8,
                      (2, 5): 8},
        }
        relevant_indices = [1, 2, 4, 5]
        assert 5 not in pegma.atom_oligomer_map

        selected = pegma.select_relevant_parameters(handler_kwargs, relevant_indices)
        atoms = pegma.atom_oligomer_map
        # expected = {
        #     "bonds": {(atoms[1], atoms[2]): pet.smirker.SingleParameter((1, 2), pegma, 3),
        #               (atoms[2], atoms[4]): pet.smirker.SingleParameter((2, 4), pegma, 8)},
        # }
        expected = {
            "bonds": [
                        pet.smirker.SingleParameter((1, 2), pegma, 3),
                        pet.smirker.SingleParameter((2, 4), pegma, 8),
                    ]
        }

        assert selected == expected

    @pytest.fixture
    def truncated_monomer(self):
        offmol = OFFMolecule.from_smiles("C/C=C/(N)C")
        monomer = pet.Monomer(offmol)
        monomer.central_atom_indices = [0, 1, 5, 6, 7, 8]
        return monomer

    @pytest.mark.parametrize("n_neighbors, n_atoms, n_bonds, n_angles, n_dihedrals, n_impropers", [
        (0, 6, 5, 7, 3, 0),
        (1, 7, 6, 9, 6, 3),
        (2, 9, 8, 12, 10, 6),
        (3, 14, 13, 21, 20, 9), 
    ])
    def test_get_central_forcefield_parameters(self, truncated_monomer, forcefield,
                                               n_neighbors, n_atoms, n_bonds,
                                               n_angles, n_dihedrals, n_impropers):
        params = truncated_monomer.get_central_forcefield_parameters(forcefield,
                                                                     n_neighbors)
        assert len(params["LibraryCharges"]) == n_atoms
        assert len(params["vdW"]) == n_atoms
        assert len(params["Bonds"]) == n_bonds
        assert len(params["Angles"]) == n_angles
        assert len(params["ProperTorsions"]) == n_dihedrals
        assert len(params["ImproperTorsions"]) == n_impropers



