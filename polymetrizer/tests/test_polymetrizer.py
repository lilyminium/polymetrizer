
import pytest
from rdkit import Chem

from polymetrizer import Polymetrizer, HYDROGEN_CAP
from openff.toolkit.topology.molecule import unit
from simtk import openmm

from polymetrizer.tests.smiles import AMINO_ACIDS_ALL_STATES, ACE, NME
from polymetrizer import Monomer
from polymetrizer.cap import Cap


class TestPolymetrizer:

    def test_forcefield_creation_nocaps(self, cys, forcefield):
        met = Polymetrizer(monomers={"Cys": cys}, caps=[HYDROGEN_CAP],
                           r_linkages={1: {2}})
        new = met.polymetrize(forcefield, include_caps=False)
        assert len(met.oligomers) == 1
        lc_handler = new.get_parameter_handler("LibraryCharges")
        assert len(lc_handler.parameters) == 1
        offmol = met.oligomers[0].to_openff().to_topology()
        matches = lc_handler.find_matches(offmol)
        assert matches
        # hydrogen won't be accounted for
        lc_handler.add_parameter(dict(smirks="[H:1][*]", charge=[0.0 * unit.elementary_charge]))

        system = openmm.System()
        lc_handler.create_force(system, offmol)
        for mol in offmol.topology_molecules:
            assert lc_handler.check_charges_assigned(mol.reference_molecule, offmol)


    def test_create_disulfide_dimers(self):
        # code from Chapin Cavender
        # SMILES for Cyx residue
        CYX = "[S]([*:3])[C]([H])([H])[C@@]([H])([C](=[O])[*:2])[N]([H])[*:1]"

        # Create monomers
        ace = Cap.from_smiles(ACE, name="Ace")
        nme = Cap.from_smiles(NME, name="Nme")
        cyx = Monomer.from_smiles(CYX, name="Cyx")
        val = Monomer.from_smiles(AMINO_ACIDS_ALL_STATES['Val'], name="Val")

        # Create Ace-Val-Cyx-Val-Nme oligomer
        ace_val_cyx_val_nme = cyx.substitute(val, r_self = 1, r_other = 2)
        ace_val_cyx_val_nme.substitute(ace, r_self = 1, r_other = 6, inplace = True)
        ace_val_cyx_val_nme.substitute(val, r_self = 2, r_other = 1, inplace = True)
        ace_val_cyx_val_nme.substitute(nme, r_self = 2, r_other = 7, inplace = True)
        assert len(ace_val_cyx_val_nme._monomer_graph) == 5

        # Create Ace-Val-Ala-Cyx-Ala-Val-Nme
        ala = Monomer.from_smiles(AMINO_ACIDS_ALL_STATES['Ala'], name="Ala")
        ace_val_ala_cyx_ala_val_nme = cyx.substitute(ala, r_self = 1, r_other = 2)
        ace_val_ala_cyx_ala_val_nme.substitute(val, r_self = 1, r_other = 2, inplace = True)
        ace_val_ala_cyx_ala_val_nme.substitute(ace, r_self = 1, r_other = 6, inplace = True)
        ace_val_ala_cyx_ala_val_nme.substitute(val, r_self = 2, r_other = 1, inplace = True)
        ace_val_ala_cyx_ala_val_nme.substitute(nme, r_self = 2, r_other = 7, inplace = True)
        assert len(ace_val_ala_cyx_ala_val_nme._monomer_graph) == 6

        disulfide = ace_val_ala_cyx_ala_val_nme.substitute(ace_val_cyx_val_nme, r_self=3, r_other=3)
        assert len(disulfide._monomer_graph) == 11
        smiles = ("[H]N(C(=O)C([H])(N([H])C(=O)C([H])(N([H])C(=O)C([H])(N([H])"
                    "C(=O)C([H])(N([H])C(=O)C([H])([H])[H])C([H])(C([H])([H])[H])"
                    "C([H])([H])[H])C([H])([H])[H])C([H])([H])SSC([H])([H])C([H])"
                    "(C(=O)N([H])C([H])(C(=O)N([H])C([H])([H])[H])C([H])(C([H])"
                    "([H])[H])C([H])([H])[H])N([H])C(=O)C([H])(N([H])C(=O)C([H])"
                    "([H])[H])C([H])(C([H])([H])[H])C([H])([H])[H])C([H])(C([H])"
                    "([H])[H])C([H])([H])[H])C([H])([H])[H]")
        assert disulfide.to_smiles(mapped=False) == smiles