from typing import Iterable, Optional, Tuple, List
import copy


from openff.toolkit.topology import Molecule as OFFMolecule


def mol_from_smiles(smiles: str) -> OFFMolecule:
    return OFFMolecule.from_smiles(smiles, allow_undefined_stereo=True)

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
        pattern = "-;!@".join([ATOM.format(i=i) for i in [3, 1, 2, 4]])
    
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