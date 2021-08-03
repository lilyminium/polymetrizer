from typing import Dict, List, Any
from collections import defaultdict
import itertools

from pydantic import PrivateAttr

from . import base


class ParameterSet:

    # TODO: should idivf even exist, it's just k by another name
    type_identifiers = ("periodicity", "phase", "idivf")  # funct for gromos

    def __init__(self, name):
        self.name = name
        self._all_parameters = defaultdict(list)
        # self._all_parameters = defaultdict(lambda: defaultdict(list))

    def add_parameters(self, parameter_set: Dict[tuple, Dict[str, Any]]):
        if isinstance(parameter_set, self):
            parameter_set = parameter_set._all_parameters

        for key, spec in parameter_set.items():
            parameter = self.get(key)
            if isinstance(spec, list):
                parameter.extend(spec)
            else:
                parameter.append(spec)

    def get(self, key):
        if key in self._all_parameters:
            return self._all_parameters[key]
        if key[::-1] in self._all_parameters:
            return self._all_parameters[key[::-1]]
        if key[-1] > key[0]:
            key = key[::-1]
        return self._all_parameters[key]

    def _keyfunc(self, item):
        return tuple(item.get(x) for x in self.type_identifiers)

    @staticmethod
    def get_keys(items=[]):
        # this also checks that there are no extra/missing keys
        if not items:
            return []
        keys = set(items[0])
        for item in items[1:]:
            assert set(item) == keys
        return keys

    def map_indices_to_graph(self, graph, mapping=None):
        if mapping is None:
            mapping = {i: node for i, node in enumerate(graph)}
        new = defaultdict(list)
        for indices, values in self._all_parameters.items():
            nodes = tuple(mapping[i] for i in indices)
            subgraph = graph.subgraph(nodes).copy()
            for i, node in enumerate(nodes):
                subgraph[node]["index"] = i
            new[subgraph].extend(values)
        self._all_parameters = new

    def average_over_keys(self, drop=[]):
        average = {}
        for atoms, specs in self._all_parameters.items():
            parameter = defaultdict(list)
            keys = [k for k in self.get_keys(specs) if k not in drop]
            sorted_ = sorted(specs, key=self._keyfunc)
            n_entries = 0
            for _, group in itertools.groupby(sorted, self._keyfunc):
                n_entries += 1
                group = list(group)
                for k in keys:
                    parameter[k].append(np.mean([item[k] for item in group], axis=0))
            if n_entries == 1:
                parameter = {k: v[0] for k, v in parameter.items()}
            average[atoms] = parameter
        return average

    # def remap_keys(self, mapping: dict = {}):
    #     new = defaultdict(list)
    #     for k, v in self._all_parameters.items():
    #         new[tuple(mapping[i] for i in k)].extend(v)
    #     self._all_parameters = new

    def filter_keys(self, keep: list = []):
        to_remove = [key for key in self._all_parameters
                     if not all(x in keep for x in key)]
        for x in to_remove:
            del self._all_parameters[x]


class ForceFieldParameterSets:

    def __init__(self, **kwargs):
        self.parameter_sets = {}
        for k, v in kwargs.items():
            pset = ParameterSet(name=k)
            pset.add_parameters(v)
            self.parameter_sets[k] = pset

    def add_parameter_set(self, values, name=None):
        if isinstance(values, ParameterSet):
            name = values.name
        if name not in self.parameter_sets:
            self.parameter_sets[name] = ParameterSet(name=name)
        pset = self.parameter_sets[name]
        pset.add_parameters(values)

    def average_over_keys(self):
        return {k: v.average_over_keys()
                for k, v in self.parameter_sets.items()}

    def map_indices_to_graph(self, graph):
        mapping = {i: node for i, node in enumerate(graph)}
        for v in self.parameter_sets.values():
            v.map_indices_to_graph(self, graph, mapping=mapping)

    # def remap_keys(self, mapping: dict = {}):
    #     for v in self.parameter_sets.values():
    #         v.remap_keys(mapping)

    def filter_keys(self, keep: list = []):
        for v in self.parameter_sets.values():
            v.filter_keys(keep)

    def __add__(self, other):
        new = type(self)
        new.__iadd__(self)
        new.__iadd__(other)
        return new

    def __iadd__(self, other):
        for k, v in other.parameter_sets.items():
            self.add_parameter_set(v, name=k)

    def __radd__(self, other):
        if other == 0:
            return self
        return other.__add__(self)
