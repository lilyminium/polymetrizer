
import pytest
from rdkit import Chem
import networkx as nx

from polymetrizer import Monomer, Oligomer

from .smiles import CYS3, CYS3PRO, CYS3PRO2


class TestOligomer:
    def test_creation_from_monomer(self, cys):
        oligomer = cys.to_oligomer()
        assert len(oligomer) == 13
        assert oligomer._constituent_monomers == {"Cys": [0]}
        assert oligomer._monomer_to_atom_nodes == {0: set(range(13))}

    def test_add(self, cys):
        pair = cys.to_oligomer()
        pair._monomer_to_atom_nodes[0] == set(range(13))
        assert pair._constituent_monomers == {"Cys": [0]}
        pair.substitute(cys, r_self=1, r_other=2, inplace=True)
        assert len(pair) == 24
        assert list(pair.graph_.nodes) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                           13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25]
        assert pair.graph.get_r_node(1) == 25
        assert pair._constituent_monomers == {"Cys": [0, 1]}
        # r_self is 13, that should
        assert pair._monomer_to_atom_nodes[0] == set(range(12))
        assert pair._monomer_to_atom_nodes[1] == {i for i in range(13, 26) if i != 22}

    @pytest.fixture()
    def multi_mol(self):
        multi_smiles = "[*:1]C([*:3])CC([*:3])([*:2])"
        return Monomer.from_smiles(multi_smiles, name="multi")

    @pytest.mark.parametrize("linkage, pairs", [
        ({1: {1}}, {1: [1]}),
        ({1: {2}}, {1: [2], 2: [1]}),
        ({1: {1, 2}}, {1: [1, 2], 2: [1]}),
        ({2: {3}}, {2: [3]}),  # two 3s doesn't register as different
    ])
    def test_map_r_substituent_pairs(self, cys_oligomer, multi_mol, linkage, pairs):
        graph = nx.Graph(linkage)
        combs_ = cys_oligomer.map_r_substituent_pairs([multi_mol], graph)
        combinations = {k: [x[0] for x in v] for k, v in combs_.items()}
        assert combinations == pairs

    def test_iter_r_group_numbers_normal(self, cys):
        assert list(cys.iter_r_group_numbers()) == [2, 1]

    def test_iter_r_group_numbers_multi(self, multi_mol):
        assert list(multi_mol.iter_r_group_numbers()) == [1, 3, 3, 2]


    @pytest.mark.parametrize("linkage, rs", [
        ({1: {1}}, [[(2, None), (1, 1)]]),
        ({1: {2}}, [[(2, 1), (1, 2)]]),
        ({1: {1, 2}}, [[(2, 1), (1, 1)], [(2, 1), (1, 2)]]),
        ({1: {2}, 2: {3}}, [[(2, 1), (1, 2)], [(2, 3), (1, 2)]]),
    ])
    def test_enumerate_substituent_combinations_single(self, cys, multi_mol, linkage, rs):
        graph = nx.Graph(linkage)
        comb = cys.enumerate_substituent_combinations([multi_mol], graph)
        gen_rs = [[(x["r_self"], x["r_other"]) for x in y] for y in comb]
        assert gen_rs == rs

    @pytest.mark.parametrize("linkage, rs", [
        ({1: {1}}, [[(2, None), (1, 1)]] * 3),
        ({1: {2}}, [[(2, 1), (1, 2)]] * 9),
        ({1: {1, 2}}, ([[(2, 1), (1, 1)]] * 3 + [[(2, 1), (1, 2)]] * 3) * 3),
        ({1: {2}, 2: {3}}, [[(2, 1), (1, 2)]] * 9 + [[(2, 3), (1, 2)]] * 3),
    ])
    def test_enumerate_substituent_combinations_multi(self, cys, pro, multi_mol, linkage, rs):
        graph = nx.Graph(linkage)
        comb = cys.enumerate_substituent_combinations([multi_mol, cys, pro], graph)
        gen_rs = [[(x["r_self"], x["r_other"]) for x in y] for y in comb]
        assert gen_rs == rs
