
import pytest
from rdkit import Chem

from polymetrizer import Polymetrizer, HYDROGEN_CAP
from openff.toolkit.topology.molecule import unit
from simtk import openmm


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
