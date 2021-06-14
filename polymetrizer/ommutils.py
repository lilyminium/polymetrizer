from collections import defaultdict

from simtk.openmm.openmm import (NonbondedForce, HarmonicBondForce,
                                 HarmonicAngleForce, PeriodicTorsionForce)

from . import utils
from .offutils import OFFParam, ChargeParam, TorsionParam


def get_nonbonded_parameters(force, **kwargs):
    charges = {}
    vdw = {}
    for fcix in range(force.getNumParticles()):
        charge, sigma, epsilon = force.getParticleParameters(int(fcix))
        charges[(fcix,)] = ChargeParam(dict(charge=[charge]))
        vdw[(fcix,)] = OFFParam(dict(sigma=sigma, epsilon=epsilon))
    return dict(LibraryCharges=charges, vdW=vdw)


def get_bond_parameters(force, **kwargs):
    parameter_names = ("length", "k")
    bonds = {}
    for i in range(force.getNumBonds()):
        param = force.getBondParameters(i)
        atoms = param[:2]
        if atoms[0] > atoms[-1]:
            atoms = atoms[::-1]
        bonds[tuple(atoms)] = OFFParam(dict(zip(parameter_names, param[2:])))
    return dict(Bonds=bonds)


def get_angle_parameters(force, **kwargs):
    parameter_names = ("angle", "k")
    angles = {}
    for i in range(force.getNumAngles()):
        param = force.getAngleParameters(i)
        atoms = param[:3]
        if atoms[0] > atoms[-1]:
            atoms = atoms[::-1]
        angles[tuple(atoms)] = OFFParam(dict(zip(parameter_names, param[3:])))
    return dict(Angles=angles)


def get_torsion_parameters(force, oligomer=None, **kwargs):
    parameter_names = ("periodicity", "phase", "k")
    propers = defaultdict(lambda: defaultdict(list))
    impropers = defaultdict(lambda: defaultdict(list))
    for i in range(force.getNumTorsions()):
        param = force.getTorsionParameters(i)
        atoms = param[:4]
        # improper: first, third atoms are bonded
        if oligomer.atoms_are_bonded(atoms[0], atoms[2]):
            atoms = [atoms[i] for i in [1, 0, 2, 3]]
            dest = impropers
        else:
            dest = propers
            if atoms[0] > atoms[-1]:
                atoms = atoms[::-1]
        atoms = tuple(atoms)
        for k, v in zip(parameter_names, param[4:]):
            dest[atoms][k].append(v)

    real_propers = {}
    real_impropers = {}
    for atoms, info in propers.items():
        real_propers[atoms] = TorsionParam(info)
    for atoms, info in impropers.items():
        real_impropers[atoms] = TorsionParam(info)
    return dict(ProperTorsions=real_propers, ImproperTorsions=real_impropers)


def quantity_to_value(obj):
    if utils.isisterable(obj):
        return [quantity_to_value(x) for x in obj]
    try:
        return obj._value
    except AttributeError:
        return obj


OPENMM_FORCE_PARSERS = {
    NonbondedForce: get_nonbonded_parameters,
    HarmonicBondForce: get_bond_parameters,
    HarmonicAngleForce: get_angle_parameters,
    PeriodicTorsionForce: get_torsion_parameters,
}
