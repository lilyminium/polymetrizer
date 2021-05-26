import pytest

import numpy as np

import polymetrizer as pet

from simtk.openmm.openmm import *
from numpy.testing import assert_equal, assert_almost_equal


from polymetrizer.offutils import OFFParam, ChargeParam, TorsionParam

def test_average_torsions(system_bta, monomer_bta):
    force = [f for f in system_bta.getForces() if isinstance(f, PeriodicTorsionForce)][0]
    torsions = pet.ommutils.get_torsion_parameters(force, oligomer=monomer_bta)

    double1 = torsions["ProperTorsions"][(0, 1, 2, 4)]
    double2 = torsions["ProperTorsions"][(1, 2, 4, 11)]
    double2.fields["phase"][1]._value = np.pi

    avg = double1.average([double1, double2])

    assert_equal(avg.fields["periodicity"], [1, 2, 3])
    assert_almost_equal([x._value for x in avg.fields["phase"]], [np.pi, np.pi, np.pi], 5)
    assert_almost_equal([x._value for x in avg.fields["k"]], [3.975604, 22.97046, 0.80180], 5)
