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
        met = pet.Polymetrizer(SPLIT_MOL_R_SMILES, r_linkages=SPLIT_MOL_R_LINKAGES)
        met.create_oligomers()
        return met

    @pytest.fixture
    def forcefield_parameters(self, forcefield, split_mol):
        return split_mol.get_forcefield_parameters(forcefield)

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

    @pytest.mark.slow
    def test_residue_based_forcefield(self, split_mol, forcefield_parameters):
        ff = split_mol._build_forcefield(forcefield_parameters, residue_based=True)
        lc_handler = ff.get_parameter_handler("LibraryCharges")
        assert len(lc_handler.parameters) == len(SPLIT_MOL_R_SMILES)
        first_smirks = ("[#7:1](-[#1:2])-[#6:3]12-[#6:4](-[#6:5]3(-[#6:6]"
                        "(-[#6:7](-[#6:8]-1(-[#1:9])-[#1:10])(-[#6:11]"
                        "(-[#6:12](-[#6:13]-3(-[#1:14])-[#1:15])(-[#6:16]-2"
                        "(-[#1:17])-[#1:18])-[#1:19])(-[#1:20])-[#1:21])-"
                        "[#1:22])(-[#1:23])-[#1:24])-[#1:25])(-[#1:26])-[#1:27]")
        assert lc_handler.parameters[0].smirks == first_smirks
        assert len(lc_handler.parameters[0].charge) == 27
    
    @pytest.mark.slow
    def test_combination_forcefield(self, split_mol, forcefield_parameters):
        ff = split_mol._build_forcefield(forcefield_parameters, residue_based=False)
        lc_handler = ff.get_parameter_handler("LibraryCharges")
        assert len(lc_handler.parameters[0].charge) == 94

    @pytest.mark.parametrize("bonds, n_mols, r_linkages", [
        ([(37, 40)], 2, {1: {2}, 2: {1}}),
        ([(15, 17), (29, 31), (43, 45), (54, 57)], 5,
         {1: {2}, 2: {1}, 3: {4}, 4: {3}, 5: {6}, 6: {5}, 7: {8}, 8: {7}})
    ])
    def test_from_offmolecule_and_bonds(self, full_offmol, bonds, n_mols, r_linkages):
        met = pet.Polymetrizer.from_offmolecule_and_bonds(full_offmol, bonds)
        assert len(met.monomers) == n_mols
        assert met.r_linkages == r_linkages
