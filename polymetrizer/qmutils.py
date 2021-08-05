# from pyscf import gto, scf
# from pyscf.geomopt.berny_solver import optimize_geometry

# def optimize_conformers(molecule):
#     from rdkit import Chem
#     rdmol = molecule.to_rdkit()
#     for i in range(rdmol.GetNumConformers()):
#         xyz = Chem.MolToXYZBlock(rdmol, confId=i)
#         spec = "\n".join(xyz.split("\n")[2:-1])
#         gtomol = gto.M(atom=spec)

# mol = gto.M(atom='N 0 0 0; N 0 0 1.2', basis='ccpvdz')
# mf = scf.RHF(mol)

BOHR_TO_ANGSTROM = 0.52917721092


def optimize_conformers(molecule, method="m06-2x/def2-TZVP"):
    from rdkit import Chem
    import psi4

    rdmol = molecule.to_rdkit()
    for i in range(rdmol.GetNumConformers()):
        xyz = Chem.MolToXYZBlock(rdmol, confId=i)
        psi4mol = psi4.core.Molecule.from_string(xyz, dtype="xyz")
        psi4.optimize_geometry(method, molecule=psi4mol)
        coords = psi4mol.geometry().np.astype("float") * BOHR_TO_ANGSTROM
        conformer = rdmol.GetConformer(i)
        for j, atom_coord in enumerate(coords):
            conformer.SetAtomPosition(j, atom_coord)
    opt = type(molecule).from_rdkit(rdmol, allow_undefined_stereo=True)
    molecule._conformers = []
    for conformer in opt._conformers:
        molecule._add_conformer(conformer)
    return molecule
