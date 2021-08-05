from typing import Set, List, Dict, Any
from collections import defaultdict
import re
import itertools

import numpy as np


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


def tuple_from_string(string):
    string = string.strip().split('(')[1].split(')')[0]
    fields = [s.strip() for s in string.split(',')]
    return tuple(map(int, fields))


def average_dicts(dicts: List[Dict[str, Any]] = []):
    if not dicts:
        return {}
    keys = set(dicts[0].keys())
    assert all(set(d.keys()) == keys for d in dicts)
    collector = defaultdict(list)
    for dct in dicts:
        for k, v in dct.items():
            if not isinstance(v, str):
                collector[k].append(v)
    n_items = len(dicts)
    return {k: np.sum(v, axis=0)/n_items for k, v in collector.items()}


def concatenate_dicts(dicts):
    keys = dicts[0].keys()
    if not all(d.keys() == keys for d in dicts):
        raise ValueError("All given dicts must have the same keys")
    collector = defaultdict(list)
    for k in keys:
        for dct in dicts:
            collector[k].append(dct[k])
    return collector


def isiterable(obj):
    """
    Returns ``True`` if ``obj`` is iterable and not a string

    Adapted from MDAnalysis.lib.util.iterable
    """
    if isinstance(obj, str):
        return False
    if hasattr(obj, "next"):
        return True
    if isinstance(obj, itertools.repeat):
        return True
    try:
        len(obj)
    except (TypeError, AttributeError):
        return False
    return True