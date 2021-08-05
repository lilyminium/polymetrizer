import pytest
from rdkit import Chem

from polymetrizer.graph import AtomGraph, MolecularGraph

from .smiles import ACE, CYS, PRO, CYS3, CYS3PRO, CYS3PRO2


@pytest.fixture()
def ace_graph():
    return MolecularGraph.from_smiles(ACE)


@pytest.fixture()
def cys_graph():
    return MolecularGraph.from_smiles(CYS)


@pytest.fixture()
def pro_graph():
    return MolecularGraph.from_smiles(PRO)


class TestMolecularGraph:

    def test_create_default_ace(self):
        # defaults: do not remove H from smiles, add missing H
        rdmol = Chem.MolFromSmiles(ACE)
        assert rdmol.GetNumAtoms() == 4
        graph = MolecularGraph.from_smiles(ACE)
        assert len(graph) == 7

    def test_create_default_cys(self):
        # defaults: do not remove H from smiles, add missing H
        rdmol = Chem.MolFromSmiles(CYS)
        assert rdmol.GetNumAtoms() == 8
        graph = MolecularGraph.from_smiles(CYS)
        assert len(graph) == 13

    def test_get_r_node_cys(self, cys_graph):
        assert cys_graph.get_r_node(1) == 13
        assert cys_graph.graph_.nodes[13]["atomic_number"] == 0
        assert cys_graph.graph_.nodes[13]["atom_map_number"] == 1
        assert cys_graph.get_r_node(2) == 10
        assert cys_graph.graph_.nodes[10]["atomic_number"] == 0
        assert cys_graph.graph_.nodes[10]["atom_map_number"] == 2

    def test_get_r_node_ace(self, ace_graph):
        assert ace_graph.get_r_node(6) == 4

    def test_pro(self, pro_graph):
        assert len(pro_graph) == 16
        assert pro_graph.get_r_node(1) == 5
        assert pro_graph.get_r_node(2) == 10

    def test_adding(self, cys_graph, pro_graph):
        cys2 = cys_graph.copy(deep=True)
        assert len(cys2) == 13
        range1 = {i: i+1 for i in range(13)}
        assert cys2.index_to_node_mapping == range1
        assert "index_to_node_mapping" in cys2.__dict__

        cys2.add(cys_graph, 13, 10)
        assert len(cys2) == 24
        assert "index_to_node_mapping" not in cys2.__dict__
        assert list(cys2.nodes) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                    14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26]
        assert cys2.get_r_node(1) == 26

        cys2.add(cys_graph, 26, 10)
        assert len(cys2) == 35
        assert cys2.to_smiles(mapped=False) == CYS3

        assert cys2.get_r_node(1) == 39
        cys2.add(pro_graph, 39, 10)
        assert len(cys2) == 49
        assert cys2.to_smiles(mapped=False) == CYS3PRO

        assert cys2.get_r_node(1) == 44
        cys2.add(pro_graph, 44, 10)
        assert len(cys2) == 63
        assert cys2.to_smiles(mapped=False) == CYS3PRO2



