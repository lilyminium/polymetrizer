from collections import defaultdict

import numpy as np
from simtk.openmm.openmm import *

from . import utils

def get_nonbonded_parameters(force):
    charges = defaultdict(list)
    vdw = defaultdict(list)
    for fcix in range(force.getNumParticles()):
        charge, sigma, epsilon = force.getParticleParameters(int(fcix))
        charges[(fcix,)] = dict(charge=[charge])
        vdw[(fcix,)] = dict(sigma=sigma, epsilon=epsilon)
    return dict(LibraryCharges=charges, vdW=vdw)

def quantity_to_value(obj):
    if utils.isisterable(obj):
        return [quantity_to_value(x) for x in obj]
    try:
        return obj._value
    except AttributeError:
        return obj

# def operate_on_quantities(func, *args, **kwargs):
#     # OpenMM's units are SO incredibly annoying
#     return func(*args, **kwargs)
    

    # accept:
    # quantities, list of quantities
    # clean = [quantity_to_value(arg) for arg in args]
    # return func(clean, **kwargs)


    # clean = []
    # for arg in args:
    #     if not isinstance(arg, list):
    #         try:
    #             arg = list(arg)
    #         except TypeError:
    #             try:
    #                 arg = np.asarray(arg)
    #             except TypeError:
    #                 arg = [arg]
    #     clean.append(arg)
    
    # clean = [np.ravel(x) for x in clean]
    # return func(*clean, **kwargs)

    # try:
    #     return func(*clean, **kwargs)
    # except TypeError:
    #     try:
    #         return func(*[x._value for x in clean], **kwargs)
    #     except (TypeError, AttributeError):
    #         clean = [[x._value for x in y] for y in clean]
    #         return func(*clean, **kwargs)


OPENMM_FORCE_PARSERS = {
    NonbondedForce: get_nonbonded_parameters
}