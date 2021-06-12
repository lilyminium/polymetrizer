import pytest

import polymetrizer as pet
from numpy.testing import assert_equal, assert_almost_equal

from polymetrizer.smirker import SingleParameter, Smirker

from .data import BMA_R_SMILES


class TestSingleParameter:

    @pytest.mark.parametrize("indices, context, compressed, smirks", [
        ((0,), "all", True,
            ("[#6:1](/[#6](=[#6](/[#7](-[#1])-[#1])-[#6]"
            "(-[#1])(-[#1])-[#1])-[#1])(-[#1])(-[#1])-[#1]")),
        ((0,), "all", False,
            ("[#6AH3X4x0!r+0:1](/[#6AH1X3x0!r+0](=[#6AH0X3x0!r+0]"
            "(/[#7AH2X3x0!r+0](-[#1AH0X1x0!r+0])-[#1AH0X1x0!r+0])-"
            "[#6AH3X4x0!r+0](-[#1AH0X1x0!r+0])(-[#1AH0X1x0!r+0])-"
            "[#1AH0X1x0!r+0])-[#1AH0X1x0!r+0])(-[#1AH0X1x0!r+0])"
            "(-[#1AH0X1x0!r+0])-[#1AH0X1x0!r+0]")),
        ((1,), "all", True,
            ("[#6](/[#6:1](=[#6](/[#7](-[#1])-[#1])-[#6]"
            "(-[#1])(-[#1])-[#1])-[#1])(-[#1])(-[#1])-[#1]")),
        ((1,), "all", False,
            ("[#6AH3X4x0!r+0](/[#6AH1X3x0!r+0:1](=[#6AH0X3x0!r+0]"
            "(/[#7AH2X3x0!r+0](-[#1AH0X1x0!r+0])-[#1AH0X1x0!r+0])-"
            "[#6AH3X4x0!r+0](-[#1AH0X1x0!r+0])(-[#1AH0X1x0!r+0])-"
            "[#1AH0X1x0!r+0])-[#1AH0X1x0!r+0])(-[#1AH0X1x0!r+0])"
            "(-[#1AH0X1x0!r+0])-[#1AH0X1x0!r+0]")),
        ((0, 1), "all", True,
            ("[#6:1](/[#6:2](=[#6](/[#7](-[#1])-[#1])-[#6]"
            "(-[#1])(-[#1])-[#1])-[#1])(-[#1])(-[#1])-[#1]")),
        ((0, 1), "all", False,
            ("[#6AH3X4x0!r+0:1](/[#6AH1X3x0!r+0:2](=[#6AH0X3x0!r+0]"
            "(/[#7AH2X3x0!r+0](-[#1AH0X1x0!r+0])-[#1AH0X1x0!r+0])-"
            "[#6AH3X4x0!r+0](-[#1AH0X1x0!r+0])(-[#1AH0X1x0!r+0])-"
            "[#1AH0X1x0!r+0])-[#1AH0X1x0!r+0])(-[#1AH0X1x0!r+0])"
            "(-[#1AH0X1x0!r+0])-[#1AH0X1x0!r+0]")),
        ((1, 0), "all", True,
            ("[#6:2](/[#6:1](=[#6](/[#7](-[#1])-[#1])-[#6]"
            "(-[#1])(-[#1])-[#1])-[#1])(-[#1])(-[#1])-[#1]")),
        ((1, 0), "all", False,
            ("[#6AH3X4x0!r+0:2](/[#6AH1X3x0!r+0:1](=[#6AH0X3x0!r+0]"
            "(/[#7AH2X3x0!r+0](-[#1AH0X1x0!r+0])-[#1AH0X1x0!r+0])-"
            "[#6AH3X4x0!r+0](-[#1AH0X1x0!r+0])(-[#1AH0X1x0!r+0])-"
            "[#1AH0X1x0!r+0])-[#1AH0X1x0!r+0])(-[#1AH0X1x0!r+0])"
            "(-[#1AH0X1x0!r+0])-[#1AH0X1x0!r+0]")),
        
        ((0,), "minimal", True, "[#6:1]"),
        ((0,), "minimal", False, "[#6AH3X4x0!r+0:1]"),
        ((1,), "minimal", True, "[#6:1]"),
        ((1,), "minimal", False, "[#6AH1X3x0!r+0:1]"),
        ((0, 1), "minimal", True, "[#6:1]-[#6:2]"),
        ((0, 1), "minimal", False, "[#6AH3X4x0!r+0:1]-[#6AH1X3x0!r+0:2]"),
        ((1, 0), "minimal", True, "[#6:2]-[#6:1]"),
        ((1, 0), "minimal", False, "[#6AH3X4x0!r+0:2]-[#6AH1X3x0!r+0:1]"),
    ])
    def test_create_smirks(self, monomer_bta, indices, context, compressed, smirks):
        param = SingleParameter(indices, monomer_bta, {})
        created = param.create_smirks(context=context, compressed=compressed)
        assert created == smirks

    @pytest.mark.parametrize("indices, smirks", [
        ((1,), ("[#6:1](-[#1])(-[#1])-[#6](-[#6](=[#8])-[#8]-[#6]"
                "(-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])"
                "-[#6](-[#1])(-[#1])-[#1])-[#6](-[#1])(-[#1])-[#1]")),
        ((2,), ("[#6](-[#1:1])(-[#1])-[#6](-[#6](=[#8])-[#8]-[#6]"
                "(-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])"
                "-[#6](-[#1])(-[#1])-[#1])-[#6](-[#1])(-[#1])-[#1]")),
        ((1, 2), ("[#6:1](-[#1:2])(-[#1])-[#6](-[#6](=[#8])-[#8]-[#6]"
                "(-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])"
                "-[#6](-[#1])(-[#1])-[#1])-[#6](-[#1])(-[#1])-[#1]")),
        ((2, 1), ("[#6:2](-[#1:1])(-[#1])-[#6](-[#6](=[#8])-[#8]-[#6]"
                "(-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])"
                "-[#6](-[#1])(-[#1])-[#1])-[#6](-[#1])(-[#1])-[#1]")),
        ((11, 12, 13), ("[#6](-[#1])(-[#1])-[#6](-[#6](=[#8])-[#8]-[#6]"
                        "(-[#1])(-[#1])-[#6:1](-[#1:2])(-[#1:3])-[#6]"
                        "(-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#1])-[#6]"
                        "(-[#1])(-[#1])-[#1]")),
    ])
    def test_create_residue_smirks(self, bma_bma, indices, smirks):
        param = SingleParameter(indices, bma_bma, {})
        created = param.create_smirks(context="residue", compressed=True)
        assert created == smirks


@pytest.fixture
def bma_bma(bma):
    bma2 = pet.Monomer(BMA_R_SMILES)
    met = pet.Polymetrizer([bma, bma2])
    met.create_oligomers(n_neighbor_monomers=0)
    return met


@pytest.fixture
def bma_bma_parameters(bma_bma, forcefield):
    parameters = bma_bma.get_forcefield_parameters(forcefield, n_overlapping_atoms=0)
    return parameters

class TestSmirker:

    def test_get_unified_smirks(self, bma_bma_parameters):
        assert len(bma_bma_parameters["Bonds"]) == 46
        smirker = Smirker(bma_bma_parameters["Bonds"])
        bonds = smirker.get_unified_smirks_parameters(context="residue", compressed=True)
        assert len(bonds) == 23

        key = ("[#6](-[#1])(-[#1])-[#6](-[#6](=[#8])-[#8]-[#6]"
               "(-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])"
               "-[#6:1](-[#1])(-[#1])-[#1:2])-[#6](-[#1])(-[#1])-[#1]")
        assert len(bonds[key].single_parameters) == 2
        assert_almost_equal(bonds[key].mean_parameter.fields["length"]._value, 0.10939, 5)
    
    def test_get_combined_smirks_parameter(self, bma_bma, bma_bma_parameters):
        lc = bma_bma_parameters["LibraryCharges"]
        assert len(lc) == 48
        smirker = Smirker(lc)
        bma = bma_bma.monomers[0]
        param = smirker.get_combined_smirks_parameter(bma, compressed=True)
        
        smirks = ("[#6:1](-[#1:2])(-[#1:3])-[#6:4](-[#6:5](=[#8:6])-[#8:7]-"
                  "[#6:8](-[#1:9])(-[#1:10])-[#6:11](-[#1:12])(-[#1:13])-"
                  "[#6:14](-[#1:15])(-[#1:16])-[#6:17](-[#1:18])(-[#1:19])-"
                  "[#1:20])-[#6:21](-[#1:22])(-[#1:23])-[#1:24]")
        assert param["smirks"] == smirks
        assert len(param["charge"]) == 24

    def test_get_smirks_for_oligomer(self, bma_bma, bma_bma_parameters):
        angles = bma_bma_parameters["Angles"]
        assert len(angles) == 80
        smirker = Smirker(angles)
        bma = bma_bma.monomers[0]
        parameters = smirker.get_smirks_for_oligomer(bma, compressed=True)
        assert len(parameters) == 40
        
        smirks = ("[#6:2](-[#1:3])(-[#1])-[#6:1](-[#6](=[#8])-[#8]-[#6]"
                  "(-[#1])(-[#1])-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#1])-"
                  "[#6](-[#1])(-[#1])-[#1])-[#6](-[#1])(-[#1])-[#1]")
        param = parameters[smirks].mean_parameter.fields
        assert_almost_equal(param["angle"]._value, 1.98369, 5)
        assert_almost_equal(param["k"]._value, 415.19503, 5)

