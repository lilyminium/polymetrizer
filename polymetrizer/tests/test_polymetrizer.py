import pytest


import polymetrizer as pet

from .data import SPLIT_MOL_R_SMILES, SPLIT_MOL_R_LINKAGES

class TestPolymetrizer:


    @pytest.mark.parametrize("given_linkages, resolved_linkages", [
        ({1: {1}}, {1: {1}}),
        ({1: [1]}, {1: {1}}),
        ({1: [2]}, {1: {2}, 2: {1}}),
        ({1: [3], 2: [3], 5: [1, 4]}, {1: {3, 5}, 2: {3}, 3: {1, 2}, 4: {5}, 5: {1, 4}})
    ])
    def test_create_r_linkages(self, given_linkages, resolved_linkages):
        met = pet.Polymetrizer(r_linkages=given_linkages)
        assert met.r_linkages == resolved_linkages

    
    @pytest.fixture
    def split_mol(self):
        return pet.Polymetrizer(SPLIT_MOL_R_SMILES, r_linkages=SPLIT_MOL_R_LINKAGES)

    def test_create_polymetrizer(self, split_mol):
        assert split_mol.r_linkages == {1: {2}, 2: {1}, 3: {4}, 4: {3}, 5: {6}, 6: {5}, 7: {8}, 8: {7}}
        caps_for_r_groups = {
            1: [(2, split_mol.monomers[1])],
            2: [(1, split_mol.monomers[0])],
            3: [(4, split_mol.monomers[2])],
            4: [(3, split_mol.monomers[1])],
            5: [(6, split_mol.monomers[3])],
            6: [(5, split_mol.monomers[2])],
            7: [(8, split_mol.monomers[4])],
            8: [(7, split_mol.monomers[3])],
        }
        assert split_mol.caps_for_r_groups == caps_for_r_groups