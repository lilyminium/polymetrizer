from . import rdutils, qmutils


def create_openmm_system(molecule, forcefield,
                         partial_charge_method: str = "am1bcc",
                         minimize_geometry: bool = True,
                         optimize_geometry: bool = True,
                         minimize_max_iter: int = 1000,
                         optimize_method: str = "m06-2x/def2-TZVP",
                         ):
    # OpenEye AM1BCC gives different charges for different atom order
    # because of the inbuilt AM1 optimization.
    chmols = []
    if minimize_geometry or optimize_geometry:
        # Let's generate a conformer and pre-minimize_geometry
        molecule.generate_conformers(n_conformers=1)
        if minimize_geometry:
            rdutils.minimize_conformers(molecule, minimize_max_iter=minimize_max_iter)
        if optimize_geometry:
            qmutils.optimize_conformers(molecule, method=optimize_method)

        # print(molecule.conformers)
        # now we need to manually generate charges
        molecule.assign_partial_charges(partial_charge_method,
                                        use_conformers=molecule.conformers)
        chmols = [molecule]

    system = forcefield.create_openmm_system(molecule.to_topology(),
                                             charge_from_molecules=chmols)
    return system
