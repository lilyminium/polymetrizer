from chemper.smirksify import SMIRKSifier
from chemper.graphs.single_graph import  SingleGraph
from chemper.graphs.cluster_graph import  ClusterGraph
import numpy as np
from collections import defaultdict

from .tkfuncs import get_chemper_mols

# def get_labeled_smirks(offmol, atom_indices, label_indices):


# def get_smirks(graph, include_indices=[]):
    


def get_parameter_smirks(atom_oligomers={}, unique_oligomers=[]):
    
    oligomer_indices = {oli : i for i, oli in enumerate(unique_oligomers)}
    n_mols = len(unique_oligomers)
    chmols = get_chemper_mols(unique_oligomers)

    atom_groups = sorted(atom_oligomers)
    atom_parameters = []
    for group in atom_groups:
        parameter_list = [[]] * n_mols
        oligomers = atom_oligomers[group]
        for oligomer in oligomers:
            index = oligomer_indices[oligomer]
            reverse_map = oligomer.reverse_atom_oligomer_map
            atom_indices = tuple(map(int, (reverse_map[i] for i in group)))
            parameter_list[index] = [atom_indices]
        atom_parameters.append((str(group), parameter_list))

    atom_smirks = {}
    smirker = SMIRKSifier(chmols, atom_parameters, strict_smirks=False, max_layers=1)
    # for atoms, (_, smirks_atom_list) in zip(atom_groups, atom_parameters):
    #     graph = ClusterGraph(chmols, smirks_atom_list, 3)
    #     atom_smirks[atoms] = graph.as_smirks(compress=False)

    # print(atom_smirks)
    # for k, v in atom_smirks.items():
    #     print(f"atoms {k}: {v}")

    for atoms, (_, smirks) in zip(atom_groups, smirker.current_smirks):
        print(smirks)
        atom_smirks[atoms] = smirks
    return atom_smirks

def get_librarycharges_param(oligomer, averaged_librarycharges):
    atoms = []
    charges = []
    for i, qualified in oligomer.atom_oligomer_map.items():
        atoms.append(int(i))
        charges.append(averaged_librarycharges[(qualified,)]["charge"][0])
    chmol = get_chemper_mols([oligomer])[0]
    graph = SingleGraph(chmol, atoms, layers="all")
    return dict(smirks=graph.as_smirks(), charge=charges)


def get_librarycharges_parameters(oligomers, averaged_librarycharges):
    smirks_params = defaultdict(list)
    for oligomer in oligomers:
        param = get_librarycharges_param(oligomer, averaged_librarycharges)
        smirks_params[param["smirks"]].append(param)
    
    params = []
    for smirks, param_list in smirks_params.items():
        charges = [p["charge"] for p in param_list]
        param = dict(smirks=smirks, charge=np.mean(charges, axis=0))
        params.append(param)
    return params