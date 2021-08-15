import pytest

from openff.toolkit.typing.engines.smirnoff import ForceField

from polymetrizer import Monomer, Cap

from .smiles import ACE, CYS, PRO, NME, ARG

# import polymetrizer as pet


@pytest.fixture(scope="session")
def forcefield():
    return ForceField("openff_unconstrained-1.3.0.offxml")


@pytest.fixture()
def ace():
    return Monomer.from_smiles(ACE, name="Ace")


@pytest.fixture()
def nme():
    return Monomer.from_smiles(NME, name="Nme")


@pytest.fixture()
def cys():
    return Monomer.from_smiles(CYS, name="Cys")

@pytest.fixture()
def arg():
    return Monomer.from_smiles(ARG, name="Arg")


@pytest.fixture()
def pro():
    return Monomer.from_smiles(PRO, name="Pro")


@pytest.fixture()
def ace_cap():
    return Cap.from_smiles(ACE, name="Ace")


@pytest.fixture()
def nme_cap():
    return Cap.from_smiles(NME, name="Nme")


@pytest.fixture()
def cys_oligomer(cys):
    return cys.to_oligomer()


@pytest.fixture()
def cys_ace_caps(cys, ace_cap):
    return cys.cap_remaining(caps=[ace_cap])


@pytest.fixture()
def cys_nme_caps(cys, nme_cap):
    return cys.cap_remaining(caps=[nme_cap])


@pytest.fixture()
def cys_ace_ffset(cys_ace_caps, forcefield):
    return cys_ace_caps.to_openff_parameterset(forcefield)


@pytest.fixture()
def cys_nme_ffset(cys_nme_caps, forcefield):
    return cys_nme_caps.to_openff_parameterset(forcefield)
