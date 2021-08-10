
import pytest
from rdkit import Chem

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

    