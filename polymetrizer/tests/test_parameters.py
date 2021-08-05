from openff.toolkit.topology import Molecule

from polymetrizer.parameters import ForceFieldParameterSets


class TestForceFieldParameterSets:

    def test_creation(self, forcefield):
        offmol = Molecule.from_smiles("CCO")
        parameters = ForceFieldParameterSets.from_openff_molecule(offmol, forcefield,
                                                                  optimize_geometry=False)
        assert len(parameters) == 6
        for each in ("LibraryCharges", "vdW", "Bonds", "Angles",
                     "ProperTorsions", "ImproperTorsions"):
            assert each in parameters
        assert len(parameters["LibraryCharges"]) == 9
        assert len(parameters["vdW"]) == 9
        assert len(parameters["Bonds"]) == 8
        assert len(parameters["Angles"]) == 13
        assert len(parameters["ProperTorsions"]) == 12
        assert len(parameters["ImproperTorsions"]) == 0

    def test_from_monomer(self, cys, forcefield):
        oligomer = cys.to_oligomer()
        oligomer.cap_remaining(inplace=True)
        assert len(oligomer) == 13
        parameters = oligomer.to_openff_parameterset(forcefield)

        assert len(parameters["LibraryCharges"]) == 12
        assert len(parameters["vdW"]) == 12
        assert len(parameters["Bonds"]) == 12
        assert len(parameters["Angles"]) == 19
        assert len(parameters["ProperTorsions"]) == 24
        assert len(parameters["ImproperTorsions"]) == 3

        n_cap_atoms = 0
        for key, value in parameters["LibraryCharges"].items():
            if key.monomer_atoms[0].cap:
                n_cap_atoms += 1
                assert len(value) == 2
            else:
                assert len(value) == 1
        assert n_cap_atoms == 1

    def test_from_oligomer(self, cys, forcefield):
        oligomer = cys.to_oligomer()
        oligomer.substitute(cys, r_self=1, r_other=2, inplace=True)
        oligomer.cap_remaining(inplace=True)
        assert len(oligomer) == 24
        parameters = oligomer.to_openff_parameterset(forcefield)

        assert len(parameters["LibraryCharges"]) == 12
        assert len(parameters["vdW"]) == 12
        assert len(parameters["Bonds"]) == 13
        assert len(parameters["Angles"]) == 23
        assert len(parameters["ProperTorsions"]) == 34
        assert len(parameters["ImproperTorsions"]) == 9

        for key, value in parameters["LibraryCharges"].items():
            assert len(value) == 2

        n_cap_angles = sum(k.contains_cap() for k in parameters["Angles"].keys())
        assert n_cap_angles == 4
        n_cap_proper = sum(k.contains_cap() for k in parameters["ProperTorsions"].keys())
        assert n_cap_proper == 6
        n_cap_improper = sum(k.contains_cap() for k in parameters["ImproperTorsions"].keys())
        assert n_cap_improper == 3
