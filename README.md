polymetrizer
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/lilyminium/polymetrizer/workflows/CI/badge.svg)](https://github.com/lilyminium/polymetrizer/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/lilyminium/polymetrizer/branch/master/graph/badge.svg)](https://codecov.io/gh/lilyminium/polymetrizer/branch/master)


Generate force fields for polymer-like molecules

e.g. to compare charges created for a whole molecule, vs. one assembled from parts:

```python
def compare_charges_between_fragmented_and_whole(forcefield, smiles):
    offmol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    print(f"Molecule has {offmol.n_atoms} atoms.")
    print("set atom map")
    offmol.properties["atom_map"] = {int(i): int(i + 1 )for i in np.arange(offmol.n_atoms)}
    pet, cleave_bonds = tzr.Polymetrizer.from_offmolecule(offmol,
                                                          return_cleaved_bonds=True)
    print("polymetrizing")
    new_ff = pet.polymetrize(ff130, fragmenter=WBOFragmenter(),
                             n_overlapping_atoms=8,
                             residue_based=True)
    print("Generated force field")
    top = offmol.to_topology()
    whole_system = forcefield.create_openmm_system(top)
    print("Created whole system")
    frag_system = new_ff.create_openmm_system(top)
    print("Created partial system")
    
    handler = new_ff.get_parameter_handler('LibraryCharges')
    omm_forcetype = handler._OPENMMTYPE
    whole_force = [fc for fc in whole_system.getForces() if type(fc) == omm_forcetype][0]
    frag_force = [fc for fc in frag_system.getForces() if type(fc) == omm_forcetype][0]
    
    whole_charges = np.zeros(offmol.n_atoms)
    frag_charges = np.zeros(offmol.n_atoms)
    
    for fcix in range(offmol.n_atoms):
        whole_charges[fcix] = whole_force.getParticleParameters(fcix)[0]._value
        frag_charges[fcix] = frag_force.getParticleParameters(fcix)[0]._value
    
    # move charge normalization here to compare
    charge = Chem.rdmolops.GetFormalCharge(offmol.to_rdkit())
    diff = charge - frag_charges.sum()
    if diff > 0:
        extrema = np.min(frag_charges)
    else:
        extrema = np.max(frag_charges)

    indices = np.where(frag_charges == extrema)[0]
    original_fragment = list(frag_charges)
    frag_charges[indices] += diff / len(indices)
    
    offmol.properties["atom_map"] = {int(i): int(i + 1 )for i in np.arange(offmol.n_atoms)}
    
    data = {
        "molecule": offmol.to_smiles(mapped=True),
        "fragments": [m.dummy_smiles for m in pet.monomers],
        "whole_charges": list(whole_charges),
        "fragment_charges": list(frag_charges),
        "original_fragment_charges": original_fragment,
        "charge_difference": diff,
        "cleave_bonds": list(map(list, cleave_bonds)),
    }
    return data
```

### Copyright

Copyright (c) 2021, Lily Wang


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.5.
