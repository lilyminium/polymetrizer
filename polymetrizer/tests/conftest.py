import pytest

from openff.toolkit.typing.engines.smirnoff import ForceField

from polymetrizer import Monomer

from .smiles import ACE, CYS, PRO

# import polymetrizer as pet


@pytest.fixture(scope="session")
def forcefield():
    return ForceField("openff_unconstrained-1.3.0.offxml")


@pytest.fixture()
def ace():
    return Monomer.from_smiles(ACE, name="Ace")


@pytest.fixture()
def cys():
    return Monomer.from_smiles(CYS, name="Cys")


@pytest.fixture()
def pro():
    return Monomer.from_smiles(PRO, name="Pro")


@pytest.fixture()
def cys_oligomer(cys):
    return cys.to_oligomer()