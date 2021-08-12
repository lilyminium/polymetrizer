import pytest

from openff.toolkit.topology import Molecule

from polymetrizer import Monomer
from polymetrizer.graph import AtomGraph
from polymetrizer.parameters import ForceFieldParameterSets

@pytest.fixture()
def cco_index_parameters(forcefield):
    offmol = Molecule.from_smiles("CCO")
    parameters = ForceFieldParameterSets.from_openff_molecule(offmol, forcefield,
                                                                optimize_geometry=False)
    return parameters

@pytest.fixture()
def cco_bonds(cco_index_parameters):
    return cco_index_parameters["Bonds"]

class TestForceFieldParameterSets:

    def test_creation(self, cco_index_parameters):
        assert len(cco_index_parameters) == 6
        for each in ("LibraryCharges", "vdW", "Bonds", "Angles",
                     "ProperTorsions", "ImproperTorsions"):
            assert each in cco_index_parameters
        assert len(cco_index_parameters["LibraryCharges"]) == 9
        assert len(cco_index_parameters["vdW"]) == 9
        assert len(cco_index_parameters["Bonds"]) == 8
        assert len(cco_index_parameters["Angles"]) == 13
        assert len(cco_index_parameters["ProperTorsions"]) == 12
        assert len(cco_index_parameters["ImproperTorsions"]) == 0

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


class TestParameterSet:
    @pytest.mark.parametrize("indices, n_bonds, n_dihedrals", [
        ([1, 2], 1, 0),
        ([4, 5, 6], 0, 0),
        ([0, 1, 2, 3, 4], 4, 2),
        ([1, 6, 7, 8], 2, 0),
    ])
    def test_filter_keys(self, cco_index_parameters, indices, n_bonds, n_dihedrals):
        assert len(cco_index_parameters["LibraryCharges"]) == 9
        assert len(cco_index_parameters["vdW"]) == 9
        assert len(cco_index_parameters["Bonds"]) == 8
        assert len(cco_index_parameters["Angles"]) == 13
        assert len(cco_index_parameters["ProperTorsions"]) == 12
        assert len(cco_index_parameters["ImproperTorsions"]) == 0

        cco_index_parameters.filter_keys(keep=indices)
        remaining = set()
        for parameter_set in cco_index_parameters.values():
            for key in parameter_set.keys():
                remaining |= set(key)
        assert remaining.issubset(set(indices))
        assert len(cco_index_parameters["LibraryCharges"]) == len(indices)
        assert len(cco_index_parameters["vdW"]) == len(indices)
        assert len(cco_index_parameters["Bonds"]) == n_bonds
        assert len(cco_index_parameters["ProperTorsions"]) == n_dihedrals

    def test_get_indices_key(self, cco_bonds):
        assert (0, 1) in cco_bonds
        assert (1, 0) not in cco_bonds
        assert cco_bonds.get((1, 0)) is cco_bonds[(0, 1)]

    def test_map_indices_to_graph(self, cco_bonds):
        cco = Monomer.from_smiles("CCO")
        assert (0, 1) in cco_bonds
        assert (1, 2) in cco_bonds
        assert (0, 3) in cco_bonds

        cco_bonds.map_indices_to_graph(cco.graph)
        assert all(isinstance(x, AtomGraph) for x in cco_bonds)
        bonds = list(cco_bonds)
        assert bonds[0].monomer_atoms[0].monomer_node == 0
        assert bonds[0].monomer_atoms[1].monomer_node == 1
        assert bonds[1].monomer_atoms[0].monomer_node == 0
        assert bonds[1].monomer_atoms[1].monomer_node == 3
        assert bonds[4].monomer_atoms[0].monomer_node == 1
        assert bonds[4].monomer_atoms[1].monomer_node == 2

    def test_add_parameters(self, cys, ace, nme):
        cys_nodes = list(cys.graph.nodes(data=True))
        for i, z in enumerate([1, 16, 6, 1, 1, 6, 1, 6, 8]):
            assert cys_nodes[i][1]["atomic_number"] == z

        cys_ace = cys.cap_remaining(caps=[ace])
        cys_nme = cys.cap_remaining(caps=[nme])
        ca_nodes = cys_ace.graph.nodes(data=True)
        cn_nodes = cys_ace.graph.nodes(data=True)
        for i, z in enumerate([1, 16, 6, 1, 1, 6, 1, 6, 8]):
            assert ca_nodes[i]["atomic_number"] == z


        
