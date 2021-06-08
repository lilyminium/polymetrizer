
import pytest

import polymetrizer as pet


@pytest.mark.parametrize("r_smiles, dummy_smiles", [
    # one R
    ("[R1]H", "[1*:1]H"),
    ("[R]H", "[*]H"),
    ("([R])H", "([*])H"),
    ("([R2])H", "([2*:2])H"),
    # mutiple R
    ("C([R1])([R2])=N([R3])", "C([1*:1])([2*:2])=N([3*:3])"),
    ("C([R])[R2]=N([R3])", "C([*])[2*:2]=N([3*:3])"),
])
def test_replace_R_and_dummy(r_smiles, dummy_smiles):
    assert pet.utils.replace_R_with_dummy(r_smiles) == dummy_smiles
    assert pet.utils.replace_dummy_with_R(dummy_smiles) == r_smiles


@pytest.mark.parametrize("dummy_smiles, wildcard_smiles", [
    # one R
    ("[1*:1]H", "[*:1]H"),
    ("[*]H", "[*]H"),
    ("([1*])H", "([*])H"),
    ("[2*]H", "[*]H"),
    # mutiple R
    ("C([1*:1])([2*:2])=N([3*:3])", "C([*:1])([*:2])=N([*:3])"),
    ("C([*])[2*]=N([3*:3])", "C([*])[*]=N([*:3])"),
])
def test_replace_dummy_with_wildcard(dummy_smiles, wildcard_smiles):
    assert pet.utils.replace_dummy_with_wildcard(dummy_smiles) == wildcard_smiles


@pytest.mark.parametrize("smiles, r_groups", [
    ("[R1]H", {1}),
    ("[R]H", set()),
    ("[1*:1]H", set()),
    ("C([R1])([R2])=N([R3])", {1, 2, 3}),
    ("C([R17])[R2]=N[R11]", {2, 11, 17})
])
def test_get_r_group_numbers_from_smiles(smiles, r_groups):
    assert pet.utils.get_r_group_numbers_from_smiles(smiles) == r_groups


@pytest.mark.parametrize("string, outtuple", [
    ("(1, 2)", (1, 2)),
    ("(2, 1)", (2, 1)),
    ("     (   1, 2) ", (1, 2)),
    (" ( 2 , 1 ) ", (2, 1))
])
def test_tuple_from_string(string, outtuple):
    assert pet.utils.tuple_from_string(string) == outtuple


