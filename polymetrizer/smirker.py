from typing import List, Tuple
from collections import defaultdict
import re
import numpy as np

from rdkit import Chem
from .tkfuncs import offmol_to_graph
from . import ommutils
from openff.toolkit.utils.toolkits import *

RDKIT_TOP_REGISTRY = ToolkitRegistry(
    toolkit_precedence=[
        RDKitToolkitWrapper,
        OpenEyeToolkitWrapper,
        AmberToolsToolkitWrapper,
        BuiltInToolkitWrapper,
    ],
    exception_if_unavailable=False,
)

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


def _get_smirks_iteration(graph, previous, root, compressed, seen):
        smirks = create_atom_smirks(graph.nodes[root], compressed)
        seen.add(previous)
        neighbors = [i for i in graph.neighbors(root) if i != previous]
        # neighbors = sorted(neighbors)
        neighbors = sorted([i for i in neighbors if i not in seen])
        n_neighbors = len(neighbors)
        for i, neighbor in enumerate(neighbors, 1):
            bond = graph.get_edge_data(root, neighbor)
            if not bond:
                bond_smirks = "~"
            else:
                bond_smirks = create_bond_smirks(bond)
            bond_smirks += _get_smirks_iteration(graph, root, neighbor, compressed, seen)
            if i < n_neighbors:
                bond_smirks = f"({bond_smirks})"
            smirks += bond_smirks
        return smirks

class ChemperGraph:
    def __init__(self, oligomer):
        self.oligomer = oligomer
        self.reverse_atom_oligomer_map = oligomer.reverse_atom_oligomer_map
        self.graph = offmol_to_graph(oligomer.offmol)
        self.rdmol = self.oligomer.offmol.to_rdkit()

    # def get_smirks(self, atom_indices=[], label_indices=[], compressed=False):
    #     for i in range(self.oligomer.offmol.n_atoms):
    #         self.graph.nodes[i]["label"] = ""
    #     for i, index in enumerate(label_indices, 1):
    #         self.graph.nodes[index]["label"] = f":{i}"

    #     indices = sorted(set(atom_indices) | set(label_indices))
    #     subgraph = self.graph.subgraph(indices)
    #     first = indices[0]
    #     seen = {first}
    #     smirks = _get_smirks_iteration(subgraph, -1, first, compressed, seen)
    #     return smirks

    def get_smirks(self, atom_indices=[], label_indices=[], compressed=False):
        for i in range(self.oligomer.offmol.n_atoms):
            self.graph.nodes[i]["label"] = ""
        for i, index in enumerate(label_indices, 1):
            self.graph.nodes[index]["label"] = f":{i}"

        rdmol = Chem.RWMol(self.rdmol)
        indices = sorted(set(atom_indices) | set(label_indices))
        for atom in rdmol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
        to_del = list(i for i in range(self.oligomer.offmol.n_atoms) if i not in indices)
        for i in to_del[::-1]:
            rdmol.RemoveAtom(i)
        rdmol.UpdatePropertyCache()
        raw_smarts = Chem.MolToSmarts(rdmol)
        smirks = raw_smarts
        PATTERN = "\[[0-9a-zA-Z#@]*:([0-9]+)]"
        for num in re.findall(PATTERN, raw_smarts):
            info = self.graph.nodes[int(num) - 1]
            atom_smirks = create_atom_smirks(info, compressed)
            new_pattern = f"\[[0-9a-zA-Z#@]*:{num}]"
            smirks = re.sub(new_pattern, atom_smirks, smirks)
        # print(raw_smarts)
        # raise ValueError()

        # subgraph = self.graph.subgraph(indices)
        # first = indices[0]
        # seen = {first}
        # smirks = _get_smirks_iteration(subgraph, -1, first, compressed, seen)
        # print(smirks)
        return smirks

    def get_qualified_indices(self, qualified_atoms):
        return [self.reverse_atom_oligomer_map[i] for i in qualified_atoms]

    def get_qualified_residue_indices(self, qualified_atoms):
        atom_indices = []
        monomers, _ = zip(*qualified_atoms)
        for (monomer, atom), index in self.reverse_atom_oligomer_map.items():
            if monomer in monomers:
                atom_indices.append(index)
        return atom_indices

    def get_qualified_smirks(self, qualified_atoms, compressed=False, return_label_indices=False):
        smirks = None
        label_indices = None
        if self.oligomer.contains_qualified_atoms(qualified_atoms):
            label_indices = self.get_qualified_indices(qualified_atoms)
            atom_indices = self.get_qualified_residue_indices(qualified_atoms)
            # if any(x in self.oligomer.central_atom_map for x in label_indices):
            # print("labels: ", label_indices)
            # print(self.oligomer.offmol.atoms[label_indices[0]].atomic_number)
            smirks = self.get_smirks(atom_indices, label_indices, compressed=compressed)
            match = self.oligomer.offmol.chemical_environment_matches(smirks, toolkit_registry=RDKIT_TOP_REGISTRY)
            assert (match and len(match[0]) == len(label_indices))
                # print(smirks)
        if return_label_indices:
            return smirks, label_indices
        return smirks

    def get_combined_smirks_param(self, parameters, compressed=False):
        # relies on 1-atom parameter!!
        counter = 1
        atoms = []
        params = []
        for qual, param in parameters.items():
            if self.oligomer.contains_qualified_atoms(qual):
                atoms.append(qual)
                params.append(param)
        flat_atoms = [x[0] for x in atoms]
        # print("flat", flat_atoms)
        # raise ValueError()
        smirks = self._get_combined_smirks(flat_atoms, compressed)
        new_param = {} # dict(smirks=smirks)
        # print(params)
        first = params[0]
        for k in first.keys():
            vals = [p[k] for p in params]
            # print("vals", vals)
            new_param[k] = np.concatenate(vals)
            # new_param[k] = ommutils.operate_on_quantities(np.concatenate, vals)
        param = Parameter(atoms, new_param)
        param.smirks.add(smirks)
        return param

        # return new_param
    
    def _get_combined_smirks(self, all_atoms, compressed=False, return_label_indices=False):
        label_indices = self.get_qualified_indices(all_atoms)
        atom_indices = np.arange(self.oligomer.offmol.n_atoms)
        smirks = self.get_smirks(atom_indices, label_indices, compressed=compressed)
        # match = self.oligomer.offmol.chemical_environment_matches(smirks)
        # assert (match and len(match[0]) == len(label_indices))
        if return_label_indices:
            return smirks, label_indices
        return smirks
            

class Parameter:
    def __init__(self, qualified_atoms, param, index=0):
        self.param = param
        self.qualified_atoms = set(qualified_atoms)
        self.smirks = set()
        self.matches = set()
        self.expected_matches = set()
        self.index = index

    def has_equal_parameters(self, other):
        a = self.param
        b = other.param
        if set(a.keys()) != set(b.keys()):
            return False
        for k in a.keys():
            ak = a[k]
            bk = b[k]
            if not ommutils.operate_on_quantities(np.allclose, ak, bk, rtol=1e-04, atol=1e-05):
                return False
        return True

    def get_smirks_and_expected_values(self, graphs, compressed=False):
        self.smirks = set()
        self.expected_matches = set()
        for atoms in self.qualified_atoms:
            for i, graph in enumerate(graphs):
                smirks, expected = graph.get_qualified_smirks(atoms, compressed=compressed, return_label_indices=True)
                if smirks is not None:
                    self.smirks.add(smirks)
                    self.expected_matches.add((i, tuple(expected)))
        # print(self.qualified_atoms, self.expected_matches, graphs[0].oligomer.atom_oligomer_map.values())
        # print(self.smirks)
        symbols = []
        for qual in self.qualified_atoms:
            symb = []
            for i, atom in qual:
                symb.append(graphs[i].oligomer.offmol.atoms[atom].atomic_number)
            symbols.append(symb)
        # print(symbols)
        # print("-----")

    def get_smirks_matches(self, smirks):
        unique_matches = set()
        for oligomer in self.oligomers:
            matches = oligomer.offmol.chemical_environment_matches(smirks)
            if not matches:
                continue
            for match in matches:
                qualified = oligomer.get_qualified_atoms(match)
                if qualified is not None:
                    unique_matches.add(qualified)
        return unique_matches
    
    def get_hierarchical_matches(self, oligomers, seen):
        self.matches = set()
        for smirks in self.smirks:
            for i, oligomer in enumerate(oligomers):
                matches = oligomer.offmol.chemical_environment_matches(smirks)
                if not matches:
                    continue
                for match in matches:
                    if all(x in oligomer.atom_oligomer_map for x in match):
                        item = (i, match)
                        if item not in seen:
                            self.matches.add(item)
                            seen.add(item)


    def __iadd__(self, other):
        self.qualified_atoms |= other.qualified_atoms
        self.smirks |= other.smirks
        self.expected_matches |= other.expected_matches
        return self

class Smirker:

    def __init__(self, oligomers, averaged_parameters, combine=False, compressed=False):
        self.oligomers = oligomers
        self.graphs = [ChemperGraph(x) for x in oligomers]
        self.averaged_parameters = {k: dict(**v) for k, v in averaged_parameters.items()}
        self.atoms_to_parameter = {}
        sorted_keys = sorted(averaged_parameters.keys())
        # print(sorted_keys)

        if not combine:
            for i, qualified_atoms in enumerate(sorted_keys):
                param = averaged_parameters[qualified_atoms]
                self.atoms_to_parameter[qualified_atoms] = Parameter([qualified_atoms], param, index=i)
            for param in self.atoms_to_parameter.values():
                param.get_smirks_and_expected_values(self.graphs, compressed=compressed)
        else:
            for graph in self.graphs:
                param = graph.get_combined_smirks_param(averaged_parameters, compressed=compressed)
                for qual in param.qualified_atoms:
                    self.atoms_to_parameter[qual] = param
        
        SMIRKS_ATOMS = defaultdict(list)
        for atoms, param in self.atoms_to_parameter.items():
            for smirks in param.smirks:
                SMIRKS_ATOMS[smirks].append(param)
        for params in SMIRKS_ATOMS.values():
            self.merge_parameters(*params)

    @property
    def sorted_parameters(self):
        unique = set(self.atoms_to_parameter.values())
        return sorted(unique, key=lambda x: x.index)
    
    def merge_parameters(self, param, *params):
        for other in params:
            param += other
        for atoms in param.qualified_atoms:
            self.atoms_to_parameter[atoms] = param
    
    def _get_hierarchical_matches(self, oligomers):
        seen = set()
        parameters = self.sorted_parameters
        max_index = parameters[-1].index + 1
        for param in parameters:
            param.get_hierarchical_matches(oligomers, seen)
        
        # print([x.index for x in parameters])
        n_extra_or_missing = 0
        
        resolved = True
        for i, param in enumerate(parameters):
            if param.matches != param.expected_matches:
                extra = param.matches - param.expected_matches
                
                qualified = [oligomers[x[0]].get_qualified_atoms(x[1]) for x in extra]
                qualified = [x for x in qualified if x is not None and x not in param.qualified_atoms]
                if qualified:
                    n_extra_or_missing += 1
                    resolved = False
                    # print(i, param.qualified_atoms, "len extra", len(qualified), qualified)
                for qual in qualified:
                    other_param = self.atoms_to_parameter[qual]
                    # print(param.param)
                    # print(other_param.param)
                    if param.has_equal_parameters(other_param):
                        self.merge_parameters(param, other_param)
                    elif other_param.smirks.intersection(param.smirks):
                        copied = dict(**param.param)
                        param.param = {k: np.mean([copied[k], other_param.param[k]], axis=0) for k in copied.keys()}
                        self.merge_parameters(param, other_param)
                        # print(other_param.smirks.intersection(param.smirks))
                        # print(param.param)
                        # print(other_param.param)
                        # raise ValueError()
                    else:
                        param.index = max_index + i
        for i, param in enumerate(parameters):
            if param.matches != param.expected_matches:
                missing = param.expected_matches - param.matches
                if missing:
                    qualified = [oligomers[x[0]].get_qualified_atoms(x[1]) for x in missing]
                    qualified = [x for x in qualified if x is not None and x not in param.qualified_atoms]
                    if qualified:
                        n_extra_or_missing += 1
                        resolved = False
                        new_index = max(self.atoms_to_parameter[x].index for x in qualified)
                        param.index = new_index + 1
                #     param.index = other_param.index
                # param.index += i
        # print(f"  num extra or missing: {n_extra_or_missing}")
        return resolved
    
    def get_hierarchical_matches(self, maxiter=10):
        resolved = False
        while not resolved and maxiter:
            # print("trying again")
            # print(len(set(self.atoms_to_parameter.values())))
            resolved = self._get_hierarchical_matches(self.oligomers)
            maxiter -= 1

        # print("\n----\n")

        # print(self.atoms_to_parameter[((5, 14),)].param)
        # print(self.atoms_to_parameter[((2, 18),)].param)
