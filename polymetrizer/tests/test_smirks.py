import copy

import pytest
from numpy.testing import assert_allclose

from rdkit import Chem

from polymetrizer import Monomer
from polymetrizer.smirks import BeSmirker, SmirkSet
from .smirks import TRP_SMIRKS
from .smiles import CYS, CYS_OCCN, ARGCYS_OCCN, CYSPRO_OCCN


@pytest.fixture()
def rdtrp():
    mon = Monomer.from_smiles("C1=CC=C2C(=C1)C(=CN2)CC(C(=O)[O-])N")
    return mon.to_rdkit()


@pytest.mark.parametrize("label_atom_hydrogen_count, indices", [
    (True, ((0, 1, 2, 3, 4, 5, 6, 7, 8,
             9, 10, 11, 12, 13, 14),)),
    (False, ((0, 1, 2, 3, 4, 5, 18, 6, 7, 8, 20,
              19, 9, 10, 11, 12, 13, 14, 24, 25,
              23, 21, 22, 17, 16, 15),)),
])
@pytest.mark.parametrize("label_atom_element", [True, False])
@pytest.mark.parametrize("label_atom_aromaticity", [True, False])
@pytest.mark.parametrize("label_atom_connectivity", [True, False])
@pytest.mark.parametrize("label_ring_connectivity", [True, False])
@pytest.mark.parametrize("label_ring_atoms", [True, False])
@pytest.mark.parametrize("label_atom_formal_charge", [True, False])
def test_besmirker(rdtrp, label_atom_element,
                   label_atom_aromaticity,
                   label_atom_hydrogen_count,
                   label_atom_connectivity,
                   label_ring_connectivity,
                   label_ring_atoms,
                   label_atom_formal_charge,
                   indices):
    smirker = BeSmirker(label_atom_element=label_atom_element,
                        label_atom_aromaticity=label_atom_aromaticity,
                        label_atom_hydrogen_count=label_atom_hydrogen_count,
                        label_atom_connectivity=label_atom_connectivity,
                        label_ring_connectivity=label_ring_connectivity,
                        label_ring_atoms=label_ring_atoms,
                        label_atom_formal_charge=label_atom_formal_charge)
    output = smirker(rdtrp, label_atom_numbers=[1])
    KEY = (label_atom_element,
           label_atom_aromaticity,
           label_atom_hydrogen_count,
           label_atom_connectivity,
           label_ring_connectivity,
           label_ring_atoms,
           label_atom_formal_charge)
    assert output == TRP_SMIRKS[KEY]

    query = Chem.MolFromSmarts(output)
    assert rdtrp.GetSubstructMatches(query) == indices


class TestSmirkSet:

    @pytest.fixture()
    def cys_nme_caps2(self, nme_cap):
        return Monomer.from_smiles(CYS).cap_remaining(caps=[nme_cap])

    @pytest.fixture()
    def ace_dih(self, cys_ace_caps):
        return cys_ace_caps.graph.atomgraph_from_indices([8, 7, 5, 9])

    @pytest.fixture()
    def nme_dih(self, cys_nme_caps2):
        return cys_nme_caps2.graph.atomgraph_from_indices([8, 7, 5, 9])

    @pytest.fixture()
    def average_capped_cys(self, cys_ace_ffset, cys_nme_caps2,
                           forcefield):
        cys_nme_ffset = cys_nme_caps2.to_openff_parameterset(forcefield)
        combined = cys_ace_ffset + cys_nme_ffset
        averaged = combined.average_over_keys()
        return averaged

    @ pytest.fixture()
    def torsions_with_duplicates(self, average_capped_cys,
                                 ace_dih, nme_dih):
        hashed = [hash(ace_dih), hash(nme_dih)]
        torsions = average_capped_cys["ProperTorsions"]
        for k in list(torsions):
            if hash(k) not in hashed:
                torsions.pop(k)
        return torsions

    @ pytest.fixture()
    def ffset_with_diff_k(self, torsions_with_duplicates, nme_dih):
        torsions = copy.deepcopy(torsions_with_duplicates)
        k = torsions[nme_dih]["k"][0]
        torsions[nme_dih]["k"][0] = k * 2
        return torsions

    @ pytest.fixture()
    def ffset_with_diff_periodicity(self, torsions_with_duplicates, nme_dih):
        torsions = copy.deepcopy(torsions_with_duplicates)
        nme_torsions = torsions[nme_dih]
        nme_torsions["periodicity"] = [2, 3]
        for kw in ("phase", "k", "idivf"):
            nme_torsions[kw] = nme_torsions[kw] * 2
        return torsions

    def test_duplicate_smarts(self, ace_dih, nme_dih,
                              torsions_with_duplicates):
        assert hash(ace_dih) != hash(nme_dih)
        assert ace_dih in torsions_with_duplicates
        assert nme_dih in torsions_with_duplicates

    def test_generate_smarts(self, cys_ace_caps,
                             cys_nme_caps2,
                             ace_dih, nme_dih,
                             torsions_with_duplicates):
        smirker = SmirkSet(average_same_smarts=False,
                           split_smarts_into_oligomer=False,
                           context="residue",
                           label_atom_formal_charge=False,
                           )
        smirker.set_compounds([cys_ace_caps, cys_nme_caps2])
        assert len(smirker.compounds) == 2
        assert smirker.generate_smarts(ace_dih) == CYS_OCCN
        assert smirker.generate_smarts(nme_dih) == CYS_OCCN

    def test_reconcile_same_parameters(self, cys_ace_caps,
                                       cys_nme_caps2,
                                       ace_dih, nme_dih,
                                       torsions_with_duplicates):
        smirker = SmirkSet(average_same_smarts=False,
                           split_smarts_into_oligomer=False,
                           context="residue",
                           label_atom_formal_charge=False)
        smirker.set_compounds([cys_ace_caps, cys_nme_caps2])
        assert len(torsions_with_duplicates) == 2
        assert ace_dih in torsions_with_duplicates
        assert nme_dih in torsions_with_duplicates
        smirker._setup_unique()
        assert len(smirker._smarts_to_parameter) == 0
        smirker._generate_initial_smarts(torsions_with_duplicates)
        assert len(smirker._smarts_to_parameter) == 1
        assert CYS_OCCN in smirker._smarts_to_parameter
        assert len(smirker._smarts_to_parameter[CYS_OCCN]) == 2
        smirker._reconcile_same_parameters()
        assert len(smirker._smarts_to_parameter) == 1
        assert CYS_OCCN in smirker._smarts_to_parameter
        assert len(smirker._smarts_to_parameter[CYS_OCCN]) == 1

    def test_generate_unique_smarts_same_parameter(self, cys_ace_caps,
                                                   cys_nme_caps2,
                                                   ace_dih, nme_dih,
                                                   torsions_with_duplicates):
        smirker = SmirkSet(average_same_smarts=False,
                           split_smarts_into_oligomer=False,
                           context="residue",
                           label_atom_formal_charge=False)
        smirker.set_compounds([cys_ace_caps, cys_nme_caps2])
        unique = smirker.generate_unique_smarts(torsions_with_duplicates)
        assert len(unique) == 1
        assert len(unique[CYS_OCCN]["k"]) == 1

    def test_generate_unique_smarts_diff_parameter_noaverage(self,
                                                             cys_ace_caps,
                                                             cys_nme_caps2,
                                                             ffset_with_diff_k):
        smirker = SmirkSet(average_same_smarts=False,
                           split_smarts_into_oligomer=False,
                           context="residue",
                           label_atom_formal_charge=False)
        smirker.set_compounds([cys_ace_caps, cys_nme_caps2])
        with pytest.raises(ValueError,
                           match="Non-unique parameters for same smarts and averaging is turned off"):
            smirker.generate_unique_smarts(ffset_with_diff_k)

    def test_generate_unique_smarts_diff_parameter_average(self,
                                                           cys_ace_caps,
                                                           cys_nme_caps2,
                                                           ffset_with_diff_k):
        smirker = SmirkSet(average_same_smarts=True,
                           split_smarts_into_oligomer=False,
                           context="residue",
                           label_atom_formal_charge=False)
        smirker.set_compounds([cys_ace_caps, cys_nme_caps2])
        unique = smirker.generate_unique_smarts(ffset_with_diff_k)
        assert len(unique) == 1
        assert len(unique[CYS_OCCN]["k"]) == 1
        assert_allclose(unique[CYS_OCCN]["k"][0]._value, -2.731356)

    def test_diff_k_same_oligomer(self,
                                  cys_ace_caps,
                                  cys_nme_caps2,
                                  ffset_with_diff_k):
        smirker = SmirkSet(average_same_smarts=False,
                           split_smarts_into_oligomer=True,
                           context="residue",
                           label_atom_formal_charge=False)
        smirker.set_compounds([cys_ace_caps, cys_nme_caps2])
        with pytest.raises(ValueError,
                           match="Could not generate smarts for all atom graphs"):
            smirker.generate_unique_smarts(ffset_with_diff_k)

    def test_diff_periodicity_same_oligomer(self,
                                            cys_ace_caps,
                                            cys_nme_caps2,
                                            ffset_with_diff_periodicity):
        smirker = SmirkSet(average_same_smarts=False,
                           split_smarts_into_oligomer=True,
                           context="residue",
                           label_atom_formal_charge=False)
        smirker.set_compounds([cys_ace_caps, cys_nme_caps2])
        with pytest.raises(ValueError,
                           match="Could not generate smarts for all atom graphs"):
            smirker.generate_unique_smarts(ffset_with_diff_periodicity)

    @ pytest.fixture()
    def cysarg(self, arg, nme_cap, forcefield):
        cys1 = Monomer.from_smiles(CYS, name="Cys")
        cysarg = cys1.substitute(arg, r_self=1, r_other=2)
        cysarg.cap_remaining(caps=[nme_cap])
        return cysarg

    @ pytest.fixture()
    def cyspro(self, pro, nme_cap, forcefield):
        cys2 = Monomer.from_smiles(CYS, name="Cys")
        cyspro = cys2.substitute(pro, r_self=1, r_other=2)
        cyspro.cap_remaining(caps=[nme_cap])
        return cyspro

    @ pytest.fixture()
    def arg_ffset(self, cysarg, forcefield):
        return cysarg.to_openff_parameterset(forcefield)

    @ pytest.fixture()
    def pro_ffset(self, cyspro, forcefield):
        return cyspro.to_openff_parameterset(forcefield)

    @ pytest.fixture()
    def arg_dih(self, cysarg):
        return cysarg.graph.atomgraph_from_indices([8, 7, 5, 9])

    @ pytest.fixture()
    def pro_dih(self, cyspro):
        return cyspro.graph.atomgraph_from_indices([8, 7, 5, 9])

    @ pytest.fixture()
    def argpro_torsions(self, arg_ffset, pro_ffset, arg_dih, pro_dih):
        combined = arg_ffset + pro_ffset
        averaged = combined.average_over_keys()
        torsions = averaged["ProperTorsions"]

        hashed = [hash(arg_dih), hash(pro_dih)]
        for dih in list(torsions):
            if hash(dih) not in hashed:
                torsions.pop(dih)
        return torsions

    def test_diff_k_diff_oligomer(self, cysarg, cyspro,
                                  arg_dih, pro_dih,
                                  argpro_torsions,):
        torsions = copy.deepcopy(argpro_torsions)
        assert arg_dih in torsions
        assert pro_dih in torsions

        torsions[pro_dih]["k"][0] = torsions[pro_dih]["k"][0] * 2
        assert torsions[pro_dih]["k"] != torsions[arg_dih]["k"]

        smirker = SmirkSet(average_same_smarts=False,
                           split_smarts_into_oligomer=True,
                           context="residue")
        smirker.set_compounds([cysarg, cyspro])
        unique = smirker.generate_unique_smarts(torsions)
        assert len(unique) == 2

        assert ARGCYS_OCCN in unique
        assert CYSPRO_OCCN in unique

        assert len(unique[ARGCYS_OCCN]["k"]) == 1
        assert_allclose(unique[ARGCYS_OCCN]["k"][0]._value, -1.820904)
        assert unique[ARGCYS_OCCN]["id"] == "ArgCys"

        assert len(unique[CYSPRO_OCCN]["k"]) == 1
        assert_allclose(unique[CYSPRO_OCCN]["k"][0]._value, -3.641808)
        assert unique[CYSPRO_OCCN]["id"] == "CysPro"

    @ pytest.fixture()
    def oligoset_with_diff_periodicity(self, pro_dih,
                                       argpro_torsions):
        torsions = copy.deepcopy(argpro_torsions)
        torsions[pro_dih]["k"] = torsions[pro_dih]["k"] * 2
        torsions[pro_dih]["periodicity"] = [2, 3]
        return torsions

    @ pytest.mark.parametrize("average", [False, True])
    def test_diff_periodicity_split(self, cysarg, cyspro,
                                    oligoset_with_diff_periodicity,
                                    average):
        smirker = SmirkSet(average_same_smarts=average,
                           split_smarts_into_oligomer=True,
                           context="residue")
        smirker.set_compounds([cysarg, cyspro])
        unique = smirker.generate_unique_smarts(oligoset_with_diff_periodicity)

        assert len(unique) == 2

        assert len(unique[ARGCYS_OCCN]["k"]) == 1
        assert_allclose(unique[ARGCYS_OCCN]["k"][0]._value, -1.820904)
        assert unique[ARGCYS_OCCN]["id"] == "ArgCys"

        assert len(unique[CYSPRO_OCCN]["k"]) == 2
        assert_allclose(unique[CYSPRO_OCCN]["k"][0]._value, -1.820904)
        assert_allclose(unique[CYSPRO_OCCN]["k"][1]._value, -1.820904)
        assert unique[CYSPRO_OCCN]["id"] == "CysPro"

    def test_generate_combined_smirks(self, cys_ace_caps, cys_nme_caps2,
                                      average_capped_cys):
        charges = average_capped_cys["LibraryCharges"]
        smirker = SmirkSet()
        smirker.set_compounds([cys_ace_caps, cys_nme_caps2])
        combined = smirker.generate_combined_smarts(charges)
        print(combined.keys())
        assert len(combined) == 1
        assert False
