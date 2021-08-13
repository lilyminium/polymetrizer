from typing import Optional, Tuple, List

import networkx as nx
from pydantic import Field

from .graph import CapGraph
from .molecule import Unit


class Cap(Unit):

    graph: CapGraph = Field(default_factory=CapGraph)
    compatible_rs: Optional[Tuple[int, ...]] = tuple()
    compatible_smiles: Optional[Tuple[str, ...]] = tuple()
    r: Optional[int] = None

    def __post_init__(self):
        if not self.name:
            self.name = "Cap"
        super().__post_init__()
        if self.r is None:
            r = self.graph_.nodes[self.graph._node].get("atom_map_number")
            if r:
                self.r = r

    

HYDROGEN_CAP = Cap.from_smiles("[R][H]", name="H")

ACE_CAP = Cap.from_smiles("CC(=O)-[R]", name="Ace",
                          compatible_smiles=["[R]N([H])C([*])C(=O)-[*]"])
NME_CAP = Cap.from_smiles("[R]NC", name="Nme",
                          compatible_smiles=["[*]N([H])C([*])C(=O)-[R]"])
