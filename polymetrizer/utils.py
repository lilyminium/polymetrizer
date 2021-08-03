from typing import Set, List, Dict, Any
from collections import defaultdict
import re
import itertools

import numpy as np

from .types import ParameterSetByAtomIndex


def replace_R_with_dummy(smiles: str):
    smiles = re.sub(r"([\\/]*)\[R([0-9]+)]", r"\1[\2*:\2]", smiles)
    smiles = re.sub(r"\[R\]", r"[*]", smiles)
    return smiles


def replace_dummy_with_R(smiles: str):
    smiles = re.sub(r"\[\*\]", r"[R]", smiles)
    return re.sub(r"\[[0-9]*\*:([0-9]+)\]", r"[R\1]", smiles)


def get_r_group_numbers_from_smiles(smiles: str) -> Set[int]:
    return set(map(int, re.findall(r"\[R([0-9]+)]", smiles)))


def replace_dummy_with_wildcard(smiles: str):
    return re.sub(r"\[[0-9]*\*(:?[0-9]*)\]", r"[*\1]", smiles)


def get_other_in_pair(self, value, pair):
    assert len(pair) == 2
    assert value in pair
    if value == pair[0]:
        return pair[1]
    return pair[0]


def filter_dictionary_by_indices(dictionary: ParameterSetByAtomIndex,
                                 indices: List[int]):
    filtered = {}
    for ix, parameter in dictionary.items():
        if np.all(np.isin(ix, indices)):
            filtered[ix] = parameter
    return filtered


def is_iterable(obj: Any) -> bool:
    """Returns ``True`` if `obj` can be iterated over and is *not* a string
    nor a :class:`NamedStream`
    .. note::

        This is adapted from MDAnalysis.lib.util.iterable.
        It is GPL licensed.
    """
    if isinstance(obj, str):
        return False
    if hasattr(obj, "__next__") or hasattr(obj, "__iter__"):
        return True
    try:
        len(obj)
    except (TypeError, AttributeError):
        return False
    return True
