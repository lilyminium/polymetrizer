import re
from functools import partial

from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx

from . import base, utils

ORDER_TO_BONDTYPE = {
    (0, False): Chem.BondType.UNSPECIFIED,
    (0, True): Chem.BondType.UNSPECIFIED,
    (1, False): Chem.BondType.SINGLE,
    (1, True): Chem.BondType.AROMATIC,
    (2, False): Chem.BondType.DOUBLE,
    (2, True): Chem.BondType.TWOANDAHALF,
    (3, False): Chem.BondType.TRIPLE,
    (3, True): Chem.BondType.THREEANDAHALF,
    (4, False): Chem.BondType.QUADRUPLE,
    (4, True): Chem.BondType.FOURANDAHALF,
    (5, False): Chem.BondType.QUINTUPLE,
    (5, True): Chem.BondType.FIVEANDAHALF,
    (6, False): Chem.BondType.HEXTUPLE,
}


def rdmol_to_nxgraph(rdmol, add_hs=True):
    graph = nx.Graph()
    if add_hs:
        rdmol = Chem.AddHs(rdmol)
    for i, atom in enumerate(rdmol.GetAtoms()):
        graph.add_node(i, atomic_number=atom.GetAtomicNum(),
                       atom_map_number=atom.GetAtomMapNum(),
                       formal_charge=atom.GetFormalCharge(),
                       is_aromatic=atom.GetIsAromatic())

    for bond in rdmol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       is_aromatic=bond.GetIsAromatic(),
                       order=int(bond.GetBondTypeAsDouble()),
                       )
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
            atom.SetAtomMapNum(node + 1)
        rwmol.AddAtom(atom)
        atom_to_ix[node] = i

    for i, j, data in graph.edges(data=True):
        a, b = atom_to_ix[i], atom_to_ix[j]
        order = int(data["order"])
        aro = bool(data["is_aromatic"])
        try:
            bondtype = ORDER_TO_BONDTYPE[(order, aro)]
        except KeyError:
            raise ValueError(f"Bond with order={order}, is_aromatic {aro} "
                             "is not currently supported")
        x = rwmol.AddBond(a, b, order=bondtype)
        # docs *say* it returns the index but it returns index + 1
        rwmol.GetBondWithIdx(x - 1).SetIsAromatic(aro)

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


def get_r_from_smiles(rdmol, smiles):    
    r = len(smiles) * 2  # probably safe
    while str(r) in smiles:
        r += 1
    smiles = utils.replace_R_with_dummy(smiles, r_number=r)
    submol = Chem.MolFromSmarts(smiles)
    for i, atom in enumerate(submol.GetAtoms()):
        if atom.GetAtomMapNum() == r:
            r_index = i
            break
    else:
        raise ValueError("Cannot find which atom represents R-group")
    r_groups = set()
    for match in rdmol.GetSubstructMatches(submol):
        potential = rdmol.GetAtomWithIdx(match[i])
        if potential.GetAtomicNum() == 0:
            r_groups.add(potential.GetIsotope())
    return r_groups
