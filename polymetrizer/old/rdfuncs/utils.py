from typing import Iterable

from rdkit import Chem
from openff.toolkit.topology import Molecule as OFFMolecule


def mol_from_smiles(smiles):
    smiles_parser = Chem.rdmolfiles.SmilesParserParams()
    smiles_parser.removeHs = False
    return Chem.MolFromSmiles(smiles, smiles_parser)


def mol_to_smarts(rdmol):
    smarts = Chem.MolToSmarts(rdmol, isomericSmiles=True)
    smarts = smarts.replace("#0", "*")
    return smarts


def mol_to_smiles(rdmol):
    return Chem.MolToSmiles(rdmol, isomericSmiles=True, allHsExplicit=True)


def assign_stereochemistry(rdmol):
    if rdmol.GetNumConformers():
        Chem.AssignStereochemistryFrom3D(rdmol)
    else:
        Chem.AssignStereochemistry(rdmol, cleanIt=True)


def mol_to_offmol(rdmol) -> OFFMolecule:
    # Chem.SanitizeMol(rdmol)
    assign_stereochemistry(rdmol)
    return OFFMolecule.from_rdkit(rdmol, allow_undefined_stereo=True)


def subset_rdmol(rdmol: Chem.Mol,
                atom_indices: Iterable[int],
                check_bonds: bool = True,
                return_atom_indices: bool = False) -> Chem.Mol:
    rdmol = Chem.RWMol(rdmol)
    to_remove = [i for i in range(rdmol.GetNumAtoms()) if i not in atom_indices]
    if check_bonds:
        multiple_bonds = []
        # check bonds
        for i in to_remove:
            atom = rdmol.GetAtomWithIdx(i)
            n_bonds = 0
            for bond in atom.GetBonds():
                other = bond.GetOtherAtomIdx(i)
                if other in atom_indices:
                    n_bonds += 1
            if n_bonds > 1:
                multiple_bonds.append(i)
        atom_indices = sorted(atom_indices + multiple_bonds)
        to_remove = [i for i in to_remove if i not in multiple_bonds]
    for i in to_remove[::-1]:
        rdmol.RemoveAtom(i)
    rdmol.UpdatePropertyCache()
    if return_atom_indices:
        return rdmol, atom_indices
    return rdmol


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
        )


def get_mol_chemper_info(oligomer):
    rdmol = oligomer.offmol.to_rdkit()
    info = {i: get_chemper_atom_info(rdmol.GetAtomWithIdx(i)) for i in oligomer.atom_oligomer_map}
    return info


def get_fragment_indices(oligomer):
    rdmol = Chem.RWMol(oligomer.offmol.to_rdkit())
    for bond in oligomer.monomer_bonds:
        rdmol.RemoveBond(*bond)
    return Chem.GetMolFrags(rdmol, asMols=False)


def create_labeled_smarts(offmol, atom_indices=[], label_indices=[]):
    rdmol = Chem.RWMol(offmol.to_rdkit())
    for num, atom in enumerate(rdmol.GetAtoms(), 1):
        atom.SetAtomMapNum(num)

    indices = set(label_indices) | set(atom_indices)
    to_del = [i for i in range(offmol.n_atoms) if i not in indices]
    for index in to_del[::-1]:
        rdmol.RemoveAtom(index)
    rdmol.UpdatePropertyCache(strict=False)
    smarts = mol_to_smarts(rdmol)
    return smarts
