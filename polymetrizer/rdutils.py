import re
from functools import partial

from rdkit import Chem
import networkx as nx

from . import base


def rdmol_to_nxgraph(rdmol, add_hs=True):
    graph = nx.Graph()
    if add_hs:
        rdmol = Chem.AddHs(rdmol)
    for i, atom in enumerate(rdmol.GetAtoms(), 1):
        graph.add_node(i, atomic_number=atom.GetAtomicNum(),
                       isotope=atom.GetIsotope(),
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
        atom.SetIsotope(data["isotope"])
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


def nxgraph_to_smarts(graph, label_nodes=[], **kwargs):
    rdmol = nxgraph_to_rdmol(mapped=True)
    return rdmol_to_smarts(rdmol, label_numbers=label_nodes, **kwargs)


def get_min_ring_size(rdatom):
    if not rdatom.IsInRing():
        return 0
    min_ring = 10000
    for i in range(min_ring):
        if rdatom.IsInRingSize(i):
            return i
    return min_ring


def get_chemper_atom_info(rdatom):
    rings = len([b for b in rdatom.GetBonds() if b.IsInRing()])

    return dict(
        atomic_number=rdatom.GetAtomicNum(),
        degree=rdatom.GetDegree(),
        connectivity=rdatom.GetTotalDegree(),
        valence=rdatom.GetTotalValence(),
        formal_charge=rdatom.GetFormalCharge(),
        hydrogen_count=rdatom.GetTotalNumHs(includeNeighbors=True),
        index=rdatom.GetIdx(),
        is_aromatic=rdatom.GetIsAromatic(),
        ring_connectivity=rings,
        is_in_ring=rdatom.IsInRing(),
        min_ring_size=get_min_ring_size(rdatom),
        label="",
    )


def atom_smarts_from_info(info, label_atom_element: bool = True,
                          label_atom_aromaticity: bool = False,
                          label_atom_hydrogen_count: bool = False,
                          label_atom_connectivity: bool = False,
                          label_ring_connectivity: bool = False,
                          label_ring_atoms: bool = False,
                          label_atom_formal_charge: bool = False):
    smarts = "*"
    if label_atom_element:
        z = info.get('atomic_number')
        smarts = f"#{z}" if z else "*"
    if label_atom_aromaticity and "is_aromatic" in info:
        aromatic = "a" if info["is_aromatic"] else "A"
        smarts += aromatic
    if label_atom_hydrogen_count and "hydrogen_count" in info:
        smarts += f"H{info['hydrogen_count']}"
    if label_atom_connectivity and "connectivity" in info:
        smarts += f"X{info['connectivity']}"
    if label_ring_connectivity and "ring_connectivity" in info:
        smarts += f"x{info['ring_connectivity']}"
    if label_ring_atoms and "min_ring_size" in info:
        ring_size = info["min_ring_size"]
        ring = f"r{ring_size}" if ring_size else "!r"
        smarts += ring
    if label_atom_formal_charge and "formal_charge" in info:
        smarts += f"{info['formal_charge']:+d}"
    return f"[{smarts}{info['label']}]"


def rdmol_to_smarts(mapped_rdmol, label_numbers=[],
                    label_ring_bonds: bool = False,
                    label_atom_element: bool = True,
                    label_atom_aromaticity: bool = False,
                    label_atom_hydrogen_count: bool = False,
                    label_atom_connectivity: bool = False,
                    label_ring_connectivity: bool = False,
                    label_ring_atoms: bool = False,
                    label_atom_formal_charge: bool = False,
                    ):
    rdmol = Chem.Mol(mapped_rdmol)
    node_info = {}
    bond_info = {}
    for bond in rdmol.GetBonds():
        atom1 = bond.GetBeginAtom().GetAtomMapNum()
        atom2 = bond.GetEndAtom().GetAtomMapNum()
        ring = "@" if bond.IsInRing() else "!@"
        bond_info[(atom1, atom2)] = ring

    for atom in rdmol.GetAtoms():
        node_info[atom.GetAtomMapNum()] = get_chemper_atom_info(atom)
        atom.SetAtomMapNum(-atom.GetAtomMapNum())

    for i, node in enumerate(label_numbers, 1):
        node_info[node]["label"] = f":{i}"

    smarts = Chem.MolToSmarts(rdmol, isomericSmiles=True)
    smarts = smarts.replace("#0", "*")

    # label bonds first
    if label_ring_bonds:
        for pair, ring in bond_info.items():
            NEW_BOND = r"\1\2" + ring + r"\3"
            for a, b in [pair, pair[::-1]]:
                OLD_BOND = (f"(\\[[0-9a-zA-Z#@]*:-{a}])"
                            "([-:=#~()]+)"
                            f"(\\[[0-9a-zA-Z#@]*:-{b}])")
                smarts = re.sub(OLD_BOND, NEW_BOND, smarts)

    # now label atoms
    smartsify = partial(atom_smarts_from_info,
                        label_atom_element=label_atom_element,
                        label_atom_aromaticity=label_atom_aromaticity,
                        label_atom_hydrogen_count=label_atom_hydrogen_count,
                        label_atom_connectivity=label_atom_connectivity,
                        label_ring_connectivity=label_ring_connectivity,
                        label_ring_atoms=label_ring_atoms,
                        label_atom_formal_charge=label_atom_formal_charge)

    for n, info in node_info.items():
        atom_smarts = smartsify(info)
        OLD_ATOM = f"\\[[0-9a-zA-Z#@]*:-{n}]"
        smarts = re.sub(OLD_ATOM, atom_smarts, smarts)

    return smarts
