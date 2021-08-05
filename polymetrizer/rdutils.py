import re
from functools import partial

from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx

from . import base


def rdmol_to_nxgraph(rdmol, add_hs=True):
    graph = nx.Graph()
    if add_hs:
        rdmol = Chem.AddHs(rdmol)
    for i, atom in enumerate(rdmol.GetAtoms(), 1):
        graph.add_node(i, atomic_number=atom.GetAtomicNum(),
                       atom_map_number=atom.GetAtomMapNum(),
                       formal_charge=atom.GetFormalCharge(),
                       is_aromatic=atom.GetIsAromatic())

    for bond in rdmol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx() + 1,
                       bond.GetEndAtomIdx() + 1,
                       is_aromatic=bond.GetIsAromatic(),
                       order=int(bond.GetBondTypeAsDouble()))
    return graph


def nxgraph_to_rdmol(graph, mapped: bool = True, sanitize: bool = True):
    rwmol = Chem.RWMol()
    atom_to_ix = {}
    for i, (node, data) in enumerate(graph.nodes(data=True)):
        atom = Chem.Atom(data["atomic_number"])
        atom.SetFormalCharge(data["formal_charge"])
        atom.SetIsotope(data["atom_map_number"])
        atom.SetIsAromatic(data["is_aromatic"])
        if mapped:
            atom.SetAtomMapNum(node)
        rwmol.AddAtom(atom)
        atom_to_ix[node] = i

    for i, j, data in graph.edges(data=True):
        a, b = atom_to_ix[i], atom_to_ix[j]
        order = Chem.BondType.values[data["order"]]
        x = rwmol.AddBond(a, b, order=order)
        rwmol.GetBondWithIdx(x - 1).SetIsAromatic(data["is_aromatic"])

    if sanitize:
        Chem.SanitizeMol(rwmol)
    return rwmol


def minimize_conformers(molecule, minimize_max_iter: int = 1000):
    rdmol = molecule.to_rdkit()
    AllChem.MMFFOptimizeMoleculeConfs(rdmol, numThreads=0,
                                      maxIters=minimize_max_iter)
    opt = type(molecule).from_rdkit(rdmol, allow_undefined_stereo=True)
    molecule._conformers = []
    for conformer in opt._conformers:
        molecule._add_conformer(conformer)
    return molecule
