from typing import Set, List, Dict, Any, Optional
from collections import defaultdict
import re
import itertools

import numpy as np


def replace_R_with_dummy(smiles: str, r_number: Optional[int] = None):
    smiles = re.sub(r"([\\/]*)\[R([0-9]+)]", r"\1[\2*:\2]", smiles)
    if r_number:
        smiles = re.sub(r"\[R\]", f"[*:{r_number}]", smiles)
    else:
        smiles = re.sub(r"\[R\]", r"[*]", smiles)
    return smiles


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
