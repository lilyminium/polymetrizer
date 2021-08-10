
import pytest

from polymetrizer import Monomer

from .smiles import CYS3, CYS3PRO, CYS3PRO2


class TestMonomer:
    def test_creation_cys(self, cys):
        assert len(cys) == 13
        assert cys.name == "Cys"
        assert all(cys.graph_.nodes[n]["central"] for n in cys.graph_)

        atoms = cys.graph.monomer_atoms
        assert all(a.monomer_name == "Cys" for a in atoms)
        zs = [a.atomic_number for a in atoms]
        assert zs == [1, 16, 6, 1, 1, 6, 1, 6, 8, 0, 7, 1, 0]


class TestAtomGraph:

    def test_create_from_moleculargraph(self, cys):
        # this is super important for making sure parameters
        # for the same atoms get grouped together
        atom_graph = cys.graph.atom_subgraph_by_indices([0, 1, 2])
        assert list(atom_graph.graph_.edges) == [(0, 1), (1, 2)]
        copy = atom_graph.copy(deep=True)
        assert hash(atom_graph.graph_) != hash(copy.graph_)
        assert hash(atom_graph) == hash(copy)
        rearranged = cys.graph.atom_subgraph_by_indices([1, 2, 0])
        assert list(rearranged.graph_.edges) == [(0, 1), (1, 2)]
        assert hash(atom_graph) != hash(rearranged)
