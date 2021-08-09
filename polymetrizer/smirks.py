import contextlib
import re
import warnings
from collections import defaultdict

from rdkit import Chem

from . import utils
from .parameters import ParameterSet


class BeSmirker:

    def __init__(self, label_atom_element: bool = True,
                 label_atom_aromaticity: bool = False,
                 label_atom_hydrogen_count: bool = False,
                 label_atom_connectivity: bool = False,
                 label_ring_connectivity: bool = False,
                 label_ring_atoms: bool = False,
                 label_atom_formal_charge: bool = False,
                 label_ring_bonds: bool = False):
        self.label_atom_element = label_atom_element
        self.label_atom_aromaticity = label_atom_aromaticity
        self.label_atom_hydrogen_count = label_atom_hydrogen_count
        self.label_atom_connectivity = label_atom_connectivity
        self.label_ring_connectivity = label_ring_connectivity
        self.label_ring_atoms = label_ring_atoms
        self.label_ring_bonds = label_ring_bonds
        self.label_atom_formal_charge = label_atom_formal_charge

    def __call__(self, rdmol, label_atom_numbers=[]):
        if not isinstance(rdmol, Chem.Mol):
            rdmol = rdmol.to_rdkit()
        rdmol = Chem.Mol(rdmol)

        node_info = {}
        bond_info = {}
        for bond in rdmol.GetBonds():
            atom1 = bond.GetBeginAtom().GetAtomMapNum()
            atom2 = bond.GetEndAtom().GetAtomMapNum()
            ring = "@" if bond.IsInRing() else "!@"
            bond_info[(atom1, atom2)] = ring

        for atom in rdmol.GetAtoms():
            node_info[atom.GetAtomMapNum()] = self.get_chemper_atom_info(atom)
            atom.SetAtomMapNum(-atom.GetAtomMapNum())

        for i, node in enumerate(label_atom_numbers, 1):
            node_info[node]["label"] = f":{i}"

        smarts = Chem.MolToSmarts(rdmol, isomericSmiles=True)
        smarts = smarts.replace("#0", "*")

        # label bonds first
        if self.label_ring_bonds:
            for pair, ring in bond_info.items():
                NEW_BOND = r"\1\2" + ring + r"\3"
                for a, b in [pair, pair[::-1]]:
                    OLD_BOND = (f"(\\[[0-9a-zA-Z#@*-+]*:-{a}])"
                                "([-:=#~()]+)"
                                f"(\\[[0-9a-zA-Z#@*-+]*:-{b}])")
                    smarts = re.sub(OLD_BOND, NEW_BOND, smarts)

        # now label atoms
        for n, info in node_info.items():
            atom_smarts = self.atom_smarts_from_info(info)
            OLD_ATOM = f"\\[[0-9a-zA-Z#@*-+]*:-{n}]"
            smarts = re.sub(OLD_ATOM, atom_smarts, smarts)

        assert ":-" not in smarts
        # smarts = re.sub(r"\[([0-9a-zA-Z#@]*):-[0-9]+]", r"[\1]", smarts)

        return smarts

    def atom_smarts_from_info(self, info):
        smarts = "*"
        if self.label_atom_element:
            z = info.get('atomic_number')
            smarts = f"#{z}" if z else "*"
        if self.label_atom_aromaticity and "is_aromatic" in info:
            aromatic = "a" if info["is_aromatic"] else "A"
            smarts += aromatic
        if self.label_atom_hydrogen_count and "hydrogen_count" in info:
            smarts += f"H{info['hydrogen_count']}"
        if self.label_atom_connectivity and "connectivity" in info:
            smarts += f"X{info['connectivity']}"
        if self.label_ring_connectivity and "ring_connectivity" in info:
            smarts += f"x{info['ring_connectivity']}"
        if self.label_ring_atoms and "min_ring_size" in info:
            ring_size = info["min_ring_size"]
            ring = f"r{ring_size}" if ring_size else "!r"
            smarts += ring
        if self.label_atom_formal_charge and "formal_charge" in info:
            smarts += f"{info['formal_charge']:+d}"
        return f"[{smarts}{info['label']}]"

    @staticmethod
    def get_chemper_atom_info(rdatom):

        def get_min_ring_size(rdatom):
            if not rdatom.IsInRing():
                return 0
            min_ring = 10000
            for i in range(min_ring):
                if rdatom.IsInRingSize(i):
                    return i
            return min_ring

        rings = len([b for b in rdatom.GetBonds() if b.IsInRing()])

        return dict(
            atomic_number=rdatom.GetAtomicNum(),
            degree=rdatom.GetDegree(),
            connectivity=rdatom.GetTotalDegree(),
            valence=rdatom.GetTotalValence(),
            formal_charge=rdatom.GetFormalCharge(),
            hydrogen_count=rdatom.GetTotalNumHs(includeNeighbors=True),
            index=rdatom.GetIdx(),
            is_aromatic=rdatom.GetIsAromatic(),
            ring_connectivity=rings,
            is_in_ring=rdatom.IsInRing(),
            min_ring_size=get_min_ring_size(rdatom),
            label="",
        )


class SmirkSet:

    def __init__(self, average_same_smarts: bool = True,
                 split_smarts_into_full: bool = True,
                 context="residue",
                 #  include_caps: bool = False,
                 **kwargs):
        self.split = split_smarts_into_full
        self.average = average_same_smarts
        self.context = context
        # self.include_caps = include_caps
        self.compounds = {}

    @contextlib.contextmanager
    def set_compounds(self, compounds):
        old_compounds = self.compounds
        self._set_compounds(compounds)
        try:
            yield self
            raise RuntimeError
        except RuntimeError:
            self.compounds = old_compounds

    def _set_compounds(self, compounds):
        self.compounds = {}
        for cpd in sorted(compounds, key=len):
            key = frozenset(cpd.monomer_atoms)
            self.compounds[key] = cpd

    def generate_smarts(self, atom_graph, return_monomer_id: bool = False):
        for cpd, nodes in self.iter_matching_subgraph_nodes(atom_graph):
            sm = cpd.to_smarts(label_nodes=nodes,
                               context=self.context,
                               #    include_caps=self.include_caps,
                               )
            if return_monomer_id:
                return sm, cpd.nodes_to_monomer_id(nodes)
            return sm
        err = f"Could not generate SMARTS for {atom_graph.monomer_atoms}"
        warnings.warn(err)

    def generate_all_smarts(self, atom_graph, return_monomer_id: bool = False):
        smarts = []
        monomer_ids = []
        for cpd, nodes in self.iter_matching_subgraph_nodes(atom_graph):
            sm, mid = cpd.to_smarts(label_nodes=nodes, context="full",
                                    return_monomer_id=True)
            monomer_ids.append(mid)
            smarts.append(sm)
        if return_monomer_id:
            return smarts, monomer_ids
        return smarts

    def iter_matching_subgraph_nodes(self, atom_graph):
        atoms = frozenset(atom_graph.monomer_atoms)
        zs = [x.atomic_number for x in atom_graph.monomer_atoms]
        for key, cpd in self.compounds.items():
            if atoms.issubset(key):
                for nodes in cpd.graph.iter_isomorphic_atom_nodes(atom_graph):
                    yield (cpd, nodes)

    def _setup_unique(self):
        self._smarts_to_parameter = defaultdict(list)
        self._smarts_to_atomgraph = defaultdict(list)
        self._smarts_to_ids = {}
        self._unique_smarts_parameters = {}

    def _generate_initial_smarts(self, parameter_set):
        for atom_graph, parameter in parameter_set.items():
            try:
                smarts, mid = self.generate_smarts(atom_graph,
                                                   return_monomer_id=True)
            except TypeError:
                continue
            self._smarts_to_parameter[smarts].append(parameter)
            self._smarts_to_atomgraph[smarts].append(atom_graph)
            self._smarts_to_ids[smarts] = mid

    def _reconcile_same_parameters(self):
        # it's ok if they have the same smarts as long
        # as the parameter values are also the same
        smarts_to_unique = defaultdict(list)
        for key, values in self._smarts_to_parameter.items():
            unique = []
            for v in values:
                try:
                    if v not in unique:
                        unique.append(v)
                except ValueError as e:
                    # it's an array
                    for k0, v0 in v.items():
                        v0 = list(v0)
                        if not any(v0 == list(xn[k0]) for xn in unique):
                            unique.append(v)
                            break
            smarts_to_unique[key] = unique
        self._smarts_to_parameter = smarts_to_unique

    def _uniquify(self, smarts, parameter_set):
        parameters = self._smarts_to_parameter[smarts]
        if len(parameters) == 1:
            parameter = dict(id=self._smarts_to_ids[smarts], **parameters[0])
            return {smarts: parameter}

        if not self.split:
            if self.average:
                pset = ParameterSet(None)
                pset.add_parameters({smarts: parameters})
                averaged = pset.average_over_keys()
                averaged[smarts]["id"] = self._smarts_to_ids[smarts]
                return averaged
            err = ("Non-unique parameters for same smarts "
                   "and averaging is turned off")
            raise ValueError(err)

        atomgraphs = self._smarts_to_atomgraph[smarts]
        full_smarts = defaultdict(list)
        full_to_ag = defaultdict(list)
        for agraph in atomgraphs:
            parameter = parameter_set[agraph]
            fulls, mids = self.generate_all_smarts(agraph,
                                                   return_monomer_id=True)
            for full, mid in zip(fulls, mids):
                full_smarts[full].append(dict(id=mid, **parameter))
                full_to_ag[full].append(agraph)

        unique = {}
        # check full smarts for duplicates
        for full, parameters in full_smarts.items():
            if len(parameters) == 1:
                unique[full] = parameters[0]
            else:
                # TODO: find the unique matches from duplicates
                # basically this is fine so long as the parameters are covered
                # by the other ones, I think
                pass
        return unique

    def generate_unique_smarts(self, parameter_set):
        assert self.context != "full"

        self._setup_unique()
        self._generate_initial_smarts(parameter_set)
        self._reconcile_same_parameters()

        unique = {}
        for smarts in self._smarts_to_parameter:
            unique.update(self._uniquify(smarts, parameter_set))
        return unique

    def generate_combined_smarts(self, parameter_set):
        # currently only valid for single atom parameters
        assert all(len(graph) == 1 for graph in parameter_set)

        atoms_to_parameters = {}
        for graph, parameter in parameter_set.items():
            atom = graph.monomer_atoms[0]
            atoms_to_parameters[atom] = parameter

        all_atoms = set(atoms_to_parameters)
        combined = {}

        for atoms, cpd in self.compounds.items():
            central_nodes = cpd.graph.get_nodes(central=True)
            cpd.graph.get_central_nodes(exclude_dummy_atoms=True)
            central_atoms = {cpd.graph_.nodes[n]["monomer_atom"]
                             for n in central_nodes}
            if central_atoms.issubset(all_atoms):
                parameter = defaultdict(list)
                for node, atom in zip(central_nodes, central_atoms):
                    for k, v in atoms_to_parameters[atom].items():
                        if utils.is_iterable(v):
                            parameter[k].extend(v)
                        else:
                            parameter[k].append(v)
                smarts = cpd.to_smarts(label_nodes=central_nodes)
                parameter["id"] = cpd.nodes_to_monomer_id(central_nodes)
                combined[smarts] = parameter
        return combined
