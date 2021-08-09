from typing import Dict, List, Any
from collections import defaultdict
import itertools
import warnings

import numpy as np
from pydantic import PrivateAttr

from . import base, ommutils, rdutils, offutils, utils


class ParameterSet:

    # TODO: should idivf even exist, it's just k by another name
    type_identifiers = ("periodicity", "phase", "idivf")  # funct for gromos

    def __init__(self, name):
        self.name = name
        self._all_parameters = defaultdict(list)

    def __len__(self):
        return len(self._all_parameters)

    def __contains__(self, item):
        return item in self._all_parameters

    def keys(self):
        return self._all_parameters.keys()

    def values(self):
        return self._all_parameters.values()

    def items(self):
        return self._all_parameters.items()

    def add_parameters(self, parameter_set: Dict[tuple, Dict[str, Any]]):
        if isinstance(parameter_set, type(self)):
            parameter_set = parameter_set._all_parameters

        for key, spec in parameter_set.items():
            parameter = self.get(key)
            if isinstance(spec, list):
                parameter.extend(spec)
            else:
                parameter.append(spec)

    def _add_parameters(self, parameter_set: Dict[tuple, Dict[str, Any]]):
        if isinstance(parameter_set, type(self)):
            parameter_set = parameter_set._all_parameters

        for key, spec in parameter_set.items():
            parameter = self._all_parameters[key]
            if isinstance(spec, list):
                parameter.extend(spec)
            else:
                parameter.append(spec)

    def get(self, key):
        if self._all_parameters:
            if key in self._all_parameters:
                return self._all_parameters[key]
            if utils.is_iterable(key):
                if key[::-1] in self._all_parameters:
                    return self._all_parameters[key[::-1]]
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

    def map_indices_to_graph(self, graph):
        new = defaultdict(list)
        for indices, values in self._all_parameters.items():
            subgraph = graph.atom_subgraph_by_indices(indices)
            new[subgraph].extend(values)
        self._all_parameters = new

    def average_over_keys(self, drop=[]):
        average = {}
        for atoms, specs in self._all_parameters.items():
            parameter = {}
            keys = [k for k in self.get_keys(specs) if k not in drop]
            sorted_ = sorted(specs, key=self._keyfunc)
            n_entries = 0
            groups = [(k, list(v)) for k, v in itertools.groupby(sorted_, self._keyfunc)]
            groups = sorted(groups, key=lambda x: len(x[1]))
            common_key, common_group = groups[-1]
            if not len(groups) == 1:
                # uh oh, this is probably a torsion where the periodicities
                # etc don't match up; e.g. [2, 3] vs [3]
                err = (f"Found multiple sets of {self.name} parameters. "
                       f"Taking the most common {common_key}")
                warnings.warn(err)
            for k in keys:
                parameter[k] = np.mean([item[k] for item in common_group], axis=0)
                if utils.is_iterable(parameter[k]):
                    parameter[k] = list(parameter[k])
            average[atoms] = parameter
        return average

    def filter_keys(self, keep: list = []):
        to_remove = [key for key in self._all_parameters
                     if not all(x in keep for x in key)]
        for x in to_remove:
            del self._all_parameters[x]


class ForceFieldParameterSets:

    @ classmethod
    def from_openmm_system(cls, system):
        bond_graph = ommutils.bond_graph_from_system(system)
        parameters = {}
        for force in system.getForces():
            try:
                parser = ommutils.OPENMM_FORCE_PARSERS[type(force)]
            except KeyError:
                continue
            else:
                parameters.update(parser(force, bond_graph=bond_graph))
        return cls(**parameters)

    @ classmethod
    def from_openff_molecule(cls, molecule, forcefield,
                             partial_charge_method: str = "am1bcc",
                             minimize_geometry: bool = True,
                             optimize_geometry: bool = False,
                             minimize_max_iter: int = 1000,
                             optimize_method: str = "m06-2x/def2-TZVP",
                             ):
        # system = offutils.create_openmm_system(molecule, forcefield,
        #                                        partial_charge_method=partial_charge_method,
        #                                        minimize_geometry=minimize_geometry,
        #                                        optimize_geometry=optimize_geometry,
        #                                        minimize_max_iter=minimize_max_iter,
        #                                        optimize_method=optimize_method)
        # pset = cls.from_openmm_system(system)
        # return pset
        smiles = "parameters/" + str(hash(molecule.to_smiles())) + ".pkl"
        try:
            import pickle
            with open(f"{smiles}", "rb") as f:
                pset = pickle.load(f)
            return pset
        except FileNotFoundError:
            import pickle
            system = offutils.create_openmm_system(molecule, forcefield,
                                                   partial_charge_method=partial_charge_method,
                                                   minimize_geometry=minimize_geometry,
                                                   optimize_geometry=optimize_geometry,
                                                   minimize_max_iter=minimize_max_iter,
                                                   optimize_method=optimize_method)
            pset = cls.from_openmm_system(system)
            try:
                with open(f"{smiles}", "wb") as f:
                    pickle.dump(pset, f)
            except:
                pass
            return pset

    def __init__(self, **kwargs):
        self.parameter_sets = {}
        for k, v in kwargs.items():
            pset = ParameterSet(name=k)
            pset.add_parameters(v)
            self.parameter_sets[k] = pset

    def __len__(self):
        return len(self.parameter_sets)

    def __contains__(self, item):
        return item in self.parameter_sets

    def keys(self):
        return self.parameter_sets.keys()

    def values(self):
        return self.parameter_sets.values()

    def items(self):
        return self.parameter_sets.items()

    def __getitem__(self, item):
        return self.parameter_sets[item]

    def add_parameter_set(self, values, name=None):
        if isinstance(values, ParameterSet):
            name = values.name
        if name not in self.parameter_sets:
            self.parameter_sets[name] = ParameterSet(name=name)
        pset = self.parameter_sets[name]
        pset._add_parameters(values)

    def average_over_keys(self):
        return {k: v.average_over_keys()
                for k, v in self.parameter_sets.items()}

    def map_indices_to_graph(self, graph):
        for v in self.parameter_sets.values():
            v.map_indices_to_graph(graph)

    def filter_keys(self, keep: list = []):
        for v in self.parameter_sets.values():
            v.filter_keys(keep)

    def __add__(self, other):
        new = type(self)()
        new.__iadd__(self)
        new.__iadd__(other)
        return new

    def __iadd__(self, other):
        for k, v in other.parameter_sets.items():
            self.add_parameter_set(v, name=k)
        return self

    def __radd__(self, other):
        if not other:
            return self
        return other.__add__(self)
