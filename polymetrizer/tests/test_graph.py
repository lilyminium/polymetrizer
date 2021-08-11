import pytest
from rdkit import Chem

from polymetrizer.graph import AtomGraph, MolecularGraph
from polymetrizer import Monomer

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

    def test_index_from_zero(self, pro_graph):
        mapping = {i: i for i in range(16)}
        assert pro_graph.index_to_node_mapping == mapping
        assert pro_graph.node_to_index_mapping == mapping

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
        assert cys_graph.get_r_node(1) == 12
        assert cys_graph.graph_.nodes[12]["atomic_number"] == 0
        assert cys_graph.graph_.nodes[12]["atom_map_number"] == 1
        assert cys_graph.get_r_node(2) == 9
        assert cys_graph.graph_.nodes[9]["atomic_number"] == 0
        assert cys_graph.graph_.nodes[9]["atom_map_number"] == 2

    def test_get_r_node_ace(self, ace_graph):
        assert ace_graph.get_r_node(6) == 3

    def test_pro(self, pro_graph):
        assert len(pro_graph) == 16
        assert pro_graph.get_r_node(1) == 4
        assert pro_graph.get_r_node(2) == 9

    def test_adding(self, cys_graph, pro_graph):
        cys2 = cys_graph.copy(deep=True)
        assert len(cys2) == 13
        range1 = {i: i for i in range(13)}
        assert cys2.index_to_node_mapping == range1
        assert "index_to_node_mapping" in cys2.__dict__

        cys2.add(cys_graph, 12, 9)
        assert len(cys2) == 24
        assert "index_to_node_mapping" not in cys2.__dict__
        assert list(cys2.nodes) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                    13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25]
        assert cys2.get_r_node(1) == 25

        cys2.add(cys_graph, 25, 9)
        assert len(cys2) == 35
        assert cys2.to_smiles(mapped=False) == CYS3

        assert cys2.get_r_node(1) == 38
        cys2.add(pro_graph, 38, 9)
        assert len(cys2) == 49
        assert cys2.to_smiles(mapped=False) == CYS3PRO

        assert cys2.get_r_node(1) == 43
        cys2.add(pro_graph, 43, 9)
        assert len(cys2) == 63
        assert cys2.to_smiles(mapped=False) == CYS3PRO2



class TestAtomGraph:

    @pytest.fixture()
    def pro(self):
        return Monomer.from_smiles(PRO, name="Pro")

    @pytest.fixture()
    def pro_atom_graph(self, pro):
        return pro.graph.atom_subgraph_by_indices([1, 2, 3, 4])

    def test_hash_depends_atom_order(self, pro_atom_graph, pro):
        copy = pro_atom_graph.copy(deep=True)
        hashed = hash(pro_atom_graph)
        assert hash(pro_atom_graph.graph_) != hash(copy.graph_)
        assert hashed == hash(copy)

        backwards = pro.graph.atom_subgraph_by_indices([4, 3, 2, 1])
        assert hashed == hash(backwards)

        jumbled = pro.graph.atom_subgraph_by_indices([1, 3, 2, 4])
        assert hashed != hash(jumbled)

    def test_hash_depends_edges(self, pro_atom_graph, pro):
        copy_graph = pro.graph.copy(deep=True)
        assert 2 not in copy_graph.graph_[3]
        copy_graph.graph_.add_edge(2, 3)
        assert 2 in copy_graph.graph_[3]

        copy_atom_graph = copy_graph.atom_subgraph_by_indices([1, 2, 3, 4])
        assert hash(copy_atom_graph) != hash(pro_atom_graph)

        # check atom order again
        backwards = copy_graph.atom_subgraph_by_indices([4, 3, 2, 1])
        assert hash(copy_atom_graph) == hash(backwards)

        jumbled = copy_graph.atom_subgraph_by_indices([4, 2, 3, 1])
        assert hash(copy_atom_graph) != hash(jumbled)

    def test_monomer_atoms(self, pro_atom_graph):
        elements = [6, 1, 7, 0]
        atoms = pro_atom_graph.monomer_atoms
        for i, (atom, z) in enumerate(zip(atoms, elements), 1):
            assert atom.atomic_number == z
            assert atom.monomer_node == i

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
