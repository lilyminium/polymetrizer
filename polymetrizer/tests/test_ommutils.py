
import pytest
import numpy as np

from numpy.testing import assert_equal, assert_almost_equal

# from simtk import unit

from simtk.openmm.openmm import *

from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.toolkit.topology import Molecule

import polymetrizer as pet


@pytest.fixture
def system(butenamine, forcefield):
    return forcefield.create_openmm_system(butenamine.to_topology())

def get_force(system, forcetype):
    for f in system.getForces():
        if isinstance(f, forcetype):
            return f

def test_get_nonbonded_parameters(system):
    force = get_force(system, NonbondedForce)
    nb = pet.ommutils.get_nonbonded_parameters(force)
    assert len(nb) == 2
    assert len(nb["LibraryCharges"]) == 14
    assert len(nb["vdW"]) == 14

    first_charge = nb["LibraryCharges"][(0,)]
    assert list(first_charge.fields.keys()) == ["charge"]
    assert len(first_charge.fields["charge"]) == 1
    assert_almost_equal(first_charge.fields["charge"][0]._value, -0.04028, decimal=5)

    last_vdw = nb["vdW"][(13,)]
    assert sorted(last_vdw.fields.keys()) == ["epsilon", "sigma"]
    assert_almost_equal(last_vdw.fields["sigma"]._value, 0.26495, decimal=5)
    assert_almost_equal(last_vdw.fields["epsilon"]._value, 0.06569, decimal=5)


def test_get_bond_parameters(system):
    force = get_force(system, HarmonicBondForce)
    bonds = pet.ommutils.get_bond_parameters(force)
    assert len(bonds) == 1
    assert len(bonds["Bonds"]) == 13
    first = bonds["Bonds"][(0, 1)]
    assert sorted(first.fields.keys()) == ["k", "length"]
    assert_almost_equal(first.fields["k"]._value, 251553.96712, 5)
    assert_almost_equal(first.fields["length"]._value, 0.15010, 5)


def test_get_angle_parameters(system):
    force = get_force(system, HarmonicAngleForce)
    angles = pet.ommutils.get_angle_parameters(force)
    assert len(angles) == 1
    assert len(angles["Angles"]) == 21
    first = angles["Angles"][(0, 1, 2)]
    assert sorted(first.fields.keys()) == ["angle", "k"]
    assert_almost_equal(first.fields["k"]._value, 583.76181, 5)
    assert_almost_equal(first.fields["angle"]._value, 2.21561, 5)


def test_get_torsion_parameters(system, monomer_bta):
    force = get_force(system, PeriodicTorsionForce)
    assert force.getNumTorsions() == 36
    torsions = pet.ommutils.get_torsion_parameters(force, oligomer=monomer_bta)
    assert len(torsions) == 2
    propers = torsions["ProperTorsions"]
    impropers = torsions["ImproperTorsions"]
    assert len(propers) == 20
    assert len(impropers) == 9

    assert sorted(impropers.keys()) == [(1, 0, 2, 8), (1, 2, 8, 0), (1, 8, 0, 2),
                                        (2, 1, 3, 4), (2, 3, 4, 1), (2, 4, 1, 3),
                                        (3, 2, 9, 10), (3, 9, 10, 2), (3, 10, 2, 9)]
    double = propers[(0, 1, 2, 4)]
    assert sorted(double.fields.keys()) == ["k", "periodicity", "phase"]
    assert_almost_equal([x._value for x in double.fields["k"]], [22.97046, 7.87510], 5)
    assert_almost_equal([x._value for x in double.fields["phase"]], [np.pi, np.pi], 5)
    assert_equal(double.fields["periodicity"], [2, 1])

    single = impropers[(3, 10, 2, 9)]
    assert sorted(single.fields.keys()) == ["k", "periodicity", "phase"]
    assert_almost_equal([x._value for x in single.fields["k"]], [1.39467], 5)
    assert_almost_equal([x._value for x in single.fields["phase"]], [np.pi], 5)
    assert_equal(single.fields["periodicity"], [2])

# # @pytest.mark.parametrize("arg, kwargs, result", [
# #     ([1, 2], {}, 1.5),
# #     (np.ones(3) * unit.elementary_charge, {}, UNIT_CHARGE),
# #     ([[UNIT_CHARGE], [UNIT_CHARGE]], {}, UNIT_CHARGE),
# #     ([[UNIT_CHARGE], [UNIT_CHARGE]], {"axis": 0}, [UNIT_CHARGE]),
# #     (np.ones((3, 1)), {"axis": 0}, [1]),
# #     (np.ones((3, 1)) * unit.elementary_charge, {"axis": 0}, [UNIT_CHARGE]),
# #     (np.ones((1, 3)), {"axis": 0}, [1, 1, 1]),
# #     (np.ones((1, 3)) * unit.elementary_charge, {"axis": 0}, [UNIT_CHARGE, UNIT_CHARGE, UNIT_CHARGE]),
# # ])
# # def test_operate_on_quantities(arg, kwargs, result):
# #     print(arg, np.mean(arg),)
# #     calc = pet.ommutils.operate_on_quantities(np.mean, arg, **kwargs)
# #     # assert_array_equal(calc, result)
# #     try:
# #         assert list(calc) == list(result)
# #     except TypeError:
# #         assert calc == result