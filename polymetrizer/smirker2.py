import itertools
from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdfuncs import offmol_to_graph, mol_to_smarts

from . import ommutils


FULL_SMIRKS = "[#{atomic_number:d}{aromaticity}H{hydrogen_count:d}X{connectivity:d}x{ring_connectivity:d}{ring_size}{formal_charge:+d}{label}]"
COMPRESSED_SMIRKS = "[#{atomic_number:d}{label}]"

def create_atom_smirks(atom_info, compressed=False):
    if compressed:
        template = COMPRESSED_SMIRKS
    else:
        template = FULL_SMIRKS

    aromaticity = "a" if atom_info["is_aromatic"] else "A"
    ring_size = atom_info["min_ring_size"]
    ring = f"r{ring_size}" if ring_size else "!r"

    data = dict(aromaticity=aromaticity, ring_size=ring, **atom_info)
    return template.format(**data)


def create_bond_smirks(bond_info):
    ring = "@" if bond_info["is_in_ring"] else "!@"
    return bond_info["order_symbol"] + ring



class ChemperGraph:

    def __init__(self, oligomer):
        self.oligomer = oligomer
        self.reverse_atom_oligomer_map = oligomer.reverse_atom_oligomer_map
        self.graph = offmol_to_graph(oligomer.offmol)
        self.rdmol = self.oligomer.offmol.to_rdkit()
    
    def create_smirks(self, atom_indices=[], label_indices=[], compressed=False):
        for i in range(self.oligomer.offmol.n_atoms):
            self.graph.nodes[i]["label"] = ""
        for i, index in enumerate(label_indices, 1):
            self.graph.nodes[index]["label"] = f":{i}"

        rdmol = Chem.RWMol(self.rdmol)
        indices = sorted(set(atom_indices) | set(label_indices))
        additional_indices = []
        for atom in rdmol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
        Chem.SanitizeMol(rdmol)

        to_del = list(i for i in range(self.oligomer.offmol.n_atoms) if i not in indices)
        for i in to_del[::-1]:
            rdmol.RemoveAtom(i)
        rdmol.UpdatePropertyCache()
        raw_smarts = mol_to_smarts(rdmol)
        smirks = raw_smarts
        PATTERN = "\[[0-9a-zA-Z#@]*:([0-9]+)]"
        for num in re.findall(PATTERN, raw_smarts):
            info = self.graph.nodes[int(num) - 1]
            atom_smirks = create_atom_smirks(info, compressed)
            new_pattern = f"\[[0-9a-zA-Z#@]*:{num}]"
            smirks = re.sub(new_pattern, atom_smirks, smirks)
        return smirks


class Parameter:

    def __init__(self, atom_wrappers, parameters):
        if not isinstance(atom_wrappers, list) or not isinstance(atom_wrappers[0], tuple):
            atom_wrappers = [atom_wrappers]
        if not isinstance(parameters, list) or not isinstance(parameters[0], dict):
            parameters = [parameters]
        self.atoms = set(atom_wrappers)
        self.parameters = list(parameters)
        self.smirks = set()
    
    def has_compatible_parameters(self, other):
        for a, b in itertools.product(self.parameters, other.parameters):
            if set(a.keys()) != set(b.keys()):
                continue
            for k in a.keys():
                ak = a[k]
                bk = b[k]
                allclose = np.allclose(ak, bk, rtol=1e-04, atol=1e-05)
                if allclose:
                    break
            else:
                return True
        return False



class Smirker:

    def __init__(self, oligomers, all_parameters={}, combine=False, compressed=True):
        self.oligomers = oligomers
        self.graphs = [ChemperGraph(x) for x in oligomers]
        self.all_parameters = {k: dict(v) for k, v in all_parameters.items()}
        self.parameters_by_atom = defaultdict(lambda: defaultdict(list))

        # for handler_name, atomkwargs in all_parameters.items():
        #     for atoms, parameters in atomkwargs.items():
        #         self.parameters_by_atom[handler_name][atoms] = 