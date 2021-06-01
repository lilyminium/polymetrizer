import numpy as np
from simtk.openmm.openmm import *

def get_nonbonded_parameters(force):
    charges = {}
    vdw = {}
    for fcix in range(force.getNumParticles()):
        charge, sigma, epsilon = force.getParticleParameters(int(fcix))
        charges[(fcix,)] = dict(charge=[charge])
        vdw[(fcix,)] = dict(sigma=sigma, epsilon=epsilon)
    return dict(LibraryCharges=charges, vdW=vdw)

def operate_on_quantities(func, *args, **kwargs):
    # OpenMM's units are SO incredibly annoying

    clean = []
    for arg in args:
        if not isinstance(arg, list):
            try:
                arg = list(arg)
            except TypeError:
                try:
                    arg = np.asarray(arg)
                except TypeError:
                    arg = [arg]
        clean.append(arg)
    
    clean = [np.ravel(x) for x in clean]

    try:
        return func(*clean, **kwargs)
    except TypeError:
        try:
            return func(*[x._value for x in clean], **kwargs)
        except (TypeError, AttributeError):
            clean = [[x._value for x in y] for y in clean]
            return func(*clean, **kwargs)


OPENMM_FORCE_PARSERS = {
    NonbondedForce: get_nonbonded_parameters
}