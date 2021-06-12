import pytest

from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.toolkit.topology import Molecule

from .data import PEGMA_R_SMILES,  BMA_R_SMILES

import polymetrizer as pet

@pytest.fixture(scope="session")
def forcefield():
    return ForceField("openff_unconstrained-1.3.0.offxml")


@pytest.fixture(scope="session")
def butenamine():
    return Molecule.from_smiles("C/C=C/(N)C")


@pytest.fixture
def monomer_bta(butenamine):
    return pet.Monomer(butenamine)

@pytest.fixture
def system_bta(butenamine, forcefield):
    return forcefield.create_openmm_system(butenamine.to_topology())


@pytest.fixture
def pegma():
    monomer = pet.Monomer(PEGMA_R_SMILES)
    assert monomer.offmol.n_atoms == 45
    assert set(monomer.r_group_indices.keys()) == {1, 2}
    assert len(monomer.atom_oligomer_map) == 43
    return monomer

@pytest.fixture
def bma():
    return pet.Monomer(BMA_R_SMILES)

@pytest.fixture
def pegma_pegma(pegma):
    return pegma.attach_substituents({1: (2, pegma)})


@pytest.fixture
def bma_bma(bma):
    return bma.attach_substituents({1: (2, bma)})
