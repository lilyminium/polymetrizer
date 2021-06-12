from typing import Iterable, Optional, Tuple, List
import copy
from collections import defaultdict

from openff.toolkit.topology import Molecule as OFFMolecule

from . import utils

def mol_from_smiles(smiles: str) -> OFFMolecule:
    return OFFMolecule.from_smiles(smiles, allow_undefined_stereo=True)

def atoms_are_bonded(atom1, atom2):
    for bond in atom1.bonds:
        other = get_other_bond_atom(bond, atom1)
        if other is atom2:
            return True
    return False

def get_other_bond_atom(bond, atom):
    if bond.atom1 is atom:
        return bond.atom2
    return bond.atom1

def get_other_bond_index(bond, index):
    if bond.atom1_index == index:
        return bond.atom2_index
    return bond.atom1_index
    
def get_bonds(
        offmol,
        pattern: Optional[str] = None,
        get_bonds_only: bool = True,
        ignore_neighbors: bool = True,
        bond_atom_numbers: Tuple[int, int] = (1, 2),
    ) -> List[tuple]:

    if pattern is None:
        # SINGLE BONDS
        ATOM = "[!$(*#*)&!$(*=*)&A&!D1:{i}]"
        pattern = "-;!@".join([ATOM.format(i=i) for i in [4, 1, 2, 3]])
    
    matches = offmol.chemical_environment_matches(pattern)
    unique_bonds = set()
    unique_matches = set()  # avoid long chains
    seen = set()

    i = bond_atom_numbers[0] - 1
    j = bond_atom_numbers[1] - 1
    for group in matches:
        if group[i] > group[j]:
            group = group[::-1]
        bond = (group[i], group[j])
        if bond not in unique_bonds:
            if not ignore_neighbors or not any(x in seen for x in group):
                unique_bonds.add(bond)
                unique_matches.add(group)
                seen |= set(group)
    if get_bonds_only:
        return unique_bonds
    return unique_matches


class OFFParam:

    def __init__(self, kwargs):
        self.fields = kwargs

    def __getitem__(self, item):
        return self.fields[item]
    
    def keys(self):
        return self.fields.keys()
    
    def values(self):
        return self.fields.values()
    
    def items(self):
        return self.fields.items()

    @classmethod
    def averager(cls, concatenated):
        return cls({k: np.mean(v) for v in concatenated.items()})

    @classmethod
    def average(cls, parameters=[]):
        fields = [p.fields for p in parameters]
        concat = utils.concatenate_dicts(fields)
        return cls.averager(concat)
        
    
class ChargeParam(OFFParam):

    @classmethod
    def averager(cls, concatenated):
        return cls({k: [np.mean(v)] for v in concatenated.items()})

class TorsionParam(OFFParam):

    @classmethod
    def average(cls, parameters=[]):
        by_periodicity_and_shift = defaultdict(list)
        for param in parameters:
            priodicities = param["periodicity"]
            shifts = param["phase"]
            ks = param["k"]

            for period, shift, k in zip(periodicities, shifts, ks):
                by_periodicity_and_shift[(period, shift._value)].append((k, shift))
        param = {"periodicity": [], "phase": [], "k": [], "idivf": []}
        # TODO: check lengths to make sure I'm not accidentally adding too many terms
        for (period, _), kshifts in by_periodicity_and_shift.items():
            ks, shifts = zip(*kshifts)
            param["periodicity"].append(period)
            param["phase"].append(shifts[0])
            param["k"].append(np.mean(ks))
            param["idivf"].append(1)

        return param
    