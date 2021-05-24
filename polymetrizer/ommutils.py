from simtk.openmm.openmm import *

def get_nonbonded_parameters(force):
    charges = {}
    vdw = {}
    for fcix in range(force.getNumParticles()):
        charge, sigma, epsilon = force.getParticleParameters(int(fcix))
        charges[(fcix,)] = dict(charge=[charge])
        vdw[(fcix,)] = dict(sigma=sigma, epsilon=epsilon)
    return dict(LibraryCharges=charges, vdW=vdw)

OPENMM_FORCE_PARSERS = {
    NonbondedForce: get_nonbonded_parameters
}