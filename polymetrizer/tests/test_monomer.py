
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

