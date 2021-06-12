import re
import itertools
from collections import defaultdict

from typing_extensions import Literal
import numpy as np


try:
    from .oefuncs import get_mol_chemper_info, get_fragment_indices, create_labeled_smarts
except ImportError:
    from .rdfuncs import get_mol_chemper_info, get_fragment_indices, create_labeled_smarts


def create_full_atom_smirks(info):
    FULL_SMIRKS = "[#{atomic_number:d}{aromaticity}H{hydrogen_count:d}X{connectivity:d}x{ring_connectivity:d}{ring_size}{formal_charge:+d}{label}]"
    aromaticity = "a" if info["is_aromatic"] else "A"
    ring_size = info["min_ring_size"]
    ring = f"r{ring_size}" if ring_size else "!r"

    data = dict(aromaticity=aromaticity, ring_size=ring, **info)
    return FULL_SMIRKS.format(**data)

def create_compressed_atom_smirks(info):
    return "[#{atomic_number:d}{label}]".format(**info)


def create_smirks(oligomer, atom_indices=[], label_indices=[], compressed=True):
    atom_chemper_info = get_mol_chemper_info(oligomer)
    raw_smarts = create_labeled_smarts(oligomer.offmol, label_indices=label_indices, atom_indices=atom_indices)
    labels = {x: f":{i}" for i, x in enumerate(label_indices, 1)}
    # substitute
    if compressed:
        func = create_compressed_atom_smirks
    else:
        func = create_full_atom_smirks

    smirks = raw_smarts

    PATTERN = "\[[0-9a-zA-Z#@]*:([0-9]+)]"
    # do a double pass
    for num in re.findall("\[[0-9a-zA-Z#@]*:([0-9]+)]", raw_smarts):
        new_pattern = f"(\[[0-9a-zA-Z#@]*:)({num})]"
        atom_smirks = r"\1-\2]"
        smirks = re.sub(new_pattern, atom_smirks, smirks)

    for num in re.findall("\[[0-9a-zA-Z#@]*:(-[0-9]+)]", smirks):
        index = (-int(num)) - 1
        info = dict(**atom_chemper_info[index])
        info["label"] = labels.get(index, "")
        atom_smirks = func(info)
        new_pattern = f"(\[[0-9a-zA-Z#@]*:)({num})]"
        smirks = re.sub(new_pattern, atom_smirks, smirks)
    return smirks


class SingleParameter:

    def __init__(self, atom_indices, oligomer, parameter):
        self.atom_indices = tuple(atom_indices)
        self.monomer_atoms = tuple(oligomer.atom_oligomer_map[i] for i in atom_indices)
        self.monomers = []
        seen = []
        for atom in self.monomer_atoms:
            if atom in seen:
                self.monomers.append(atom.monomer)
            else:
                seen.append(atom)
                if atom.monomer not in self.monomers:
                    self.monomers.append(atom.monomer)
        self.oligomer = oligomer
        self.parameter = parameter
        # TODO: terribly inefficient -- make a FrozenOligomer or something

    def __repr__(self):
        clsname = type(self).__name__
        return f"{clsname}(atom_indices={self.atom_indices}, oligomer={self.oligomer}, parameter={self.parameter})"

    def __eq__(self, other):
        return (self.atom_indices, self.oligomer, self.parameter) == (other.atom_indices, other.oligomer, other.parameter)

    def create_smirks(self, context: Literal["all", "residue", "minimal"]="residue", compressed=True):
        if context == "all":
            indices = list(self.oligomer.atom_oligomer_map)
        elif context == "minimal":
            indices = list(self.atom_indices)
        elif context == "residue":
            if len(self.monomers) == 1:  # easy case
                tmp = type(self)([a.index for a in self.monomer_atoms], self.monomers[0], self.parameter)
                return tmp.create_smirks(context="all", compressed=compressed)
            fragment_indices = get_fragment_indices(self.oligomer)
            indices = []
            for fragment in fragment_indices:
                if any(i in fragment for i in self.atom_indices):
                    indices.extend(fragment)
        
        return create_smirks(self.oligomer, atom_indices=indices, label_indices=self.atom_indices,
                             compressed=compressed)
        
        smirks = create_labeled_smarts(self.oligomer.offmol, indices, self.atom_indices)
        return smirks

    def is_compatible_with(self, other):
        if self.parameter.keys() != other.parameter.keys():
            return False
        for k in self.parameter.keys():
            ak = self.parameter[k]
            bk = other.parameter[k]
            # if not np.allclose(ak, bk, rtol=1e-04, atol=1e-05):
            #     return False
        return True

    def copy_with_parameter(self, parameter):
        new = type(self)(self.atom_indices, self.oligomer, parameter)
        return new
        



class AtomGroupParameter:

    def __init__(self, single_parameters=[]):
        self.single_parameters = list(single_parameters)

    @property
    def monomer_atoms(self):
        return set(p.monomer_atoms for p in self.single_parameters)

    @property
    def parameters(self):
        return [p.parameter for p in self.single_parameters]

    @property
    def mean_parameter(self):
        parameters = self.parameters
        first = parameters[0]
        return first.average(parameters)



def are_parameters_compatible(parameters):
    parameters = list(parameters)
    compatible = [parameters.pop(0)]
    while parameters:
        current = parameters.pop(0)
        if any(current.is_compatible_with(other) for other in compatible):
            compatible.append(current)
        else:
            return False
    return True


class Smirker:

    def __init__(self, parameters_by_monomer_atoms={}, handler_name=None):
        self.handler_name = handler_name
        # average all parameters that span the same monomer atoms
        self.single_parameters = []
        self.parameters_by_monomer_atoms = {}
        for monomer_atoms, single_parameters in parameters_by_monomer_atoms.items():
            grouped = AtomGroupParameter(single_parameters)
            self.parameters_by_monomer_atoms[monomer_atoms] = grouped
            avg = grouped.mean_parameter
            for single in single_parameters:
                self.single_parameters.append(single.copy_with_parameter(avg))

        self._is_singular_atom_term = all(len(p.atom_indices) == 1 for p in self.single_parameters)
    
    def get_unified_smirks_parameters(self, context: Literal["all", "residue", "minimal"]="residue", compressed=True):
        smirks_to_param = defaultdict(list)

        # identify parameters with identical smirks
        for parameter in self.single_parameters:
            smirk = parameter.create_smirks(context=context, compressed=compressed)
            smirks_to_param[smirk].append(parameter)
        
        # merge if compatible
        return self._unify_smirks(smirks_to_param)

    @staticmethod
    def _unify_smirks(smirks_to_param):
        unified_smirks = {}
        for smirks, params in smirks_to_param.items():
            err = "I don't know how to deal with identical smirks for different parameters yet"
            assert are_parameters_compatible(params), err
            unified_smirks[smirks] = AtomGroupParameter(params)
        return unified_smirks

    def get_smirks_for_oligomer(self, oligomer, compressed=True):
        smirks_to_param = defaultdict(list)

        for monomer_atoms, group_parameter in self.parameters_by_monomer_atoms.items():
            avg = group_parameter.mean_parameter
            _, all_label_indices = oligomer.contains_monomer_atoms(monomer_atoms, handler_name=self.handler_name,
                                                                   return_indices=True)
            for label_indices in all_label_indices:
                tmp_parameter = SingleParameter(label_indices, oligomer, avg)
                smirks = tmp_parameter.create_smirks(context="all", compressed=compressed)
                smirks_to_param[smirks].append(tmp_parameter)
        
        return self._unify_smirks(smirks_to_param)



    def get_combined_smirks_parameter(self, oligomer, compressed=True):
        if not self._is_singular_atom_term:
            raise NotImplementedError("Combined SMIRKS is only supported for 1-atom terms")

        label_indices = []
        parameters = defaultdict(list)
        # parameters = []
        for i, wrapper in oligomer.atom_oligomer_map.items():
            try:
                param = self.parameters_by_monomer_atoms[(wrapper,)]
            except KeyError:
                pass
            else:
                label_indices.append(i)
                for k, v in param.mean_parameter.items():
                    parameters[k].append(v)

        n_indices = len(label_indices)
        all_indices = [i for i, atom in enumerate(oligomer.offmol.atoms) if atom.atomic_number != 0]
        smirks = create_smirks(oligomer, atom_indices=all_indices, label_indices=label_indices, compressed=compressed)
        
        flat = {}
        for k, v in parameters.items():
            flat[k] = np.array(v).reshape((-1,))
            if not len(flat[k]) == n_indices:
                raise ValueError("length mismatch between parameters and indices: "
                                 f"len(indices) == {n_indices}, len({k}) == {len(flat[k])}")
        flat["smirks"] = smirks
        return flat

