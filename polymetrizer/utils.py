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


def get_other_in_pair(self, value, pair):
    assert len(pair) == 2
    assert value in pair
    if value == pair[0]:
        return pair[1]
    return pair[0]


def is_iterable(obj: Any) -> bool:
    import networkx as nx
    if isinstance(obj, (str, nx.Graph)):
        return False
    # this rules out Quantities, the _bane of my life_
    try:
        for x in obj:
            pass
    except TypeError:
        return False
    except AttributeError:
        pass
    else:
        return True
    if hasattr(obj, "__next__"):
        return True
    return False


class cached_property:
    # this is a crappier version of the functools one

    def __init__(self, func):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__

    def __set_name__(self, cls, name):
        self.attrname = name
        self.cls = cls
        self.qualname = f"{cls.__name__}.{self.attrname}"

    def __get__(self, instance, cls=None):
        if instance is None:
            return self
        try:
            value = instance.__dict__[self.attrname]
        except KeyError:
            value = self.func(instance)
            instance.__dict__[self.attrname] = value
        return value

    def __set__(self, instance, value):
        if value is not None:
            raise ValueError(f"Cannot set {self.qualname} property")
        instance.__dict__.pop(self.attrname, None)

    def clear(self, instance):
        instance.__dict__.pop(self.attrname, None)


def uncache_properties(*propnames, pre=False):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if pre:
                self.uncache(*propnames)
            val = func(self, *args, **kwargs)
            if not pre:
                self.uncache(*propnames)
            return val
        return wrapper
    return decorator
