from typing import List, Tuple, Dict, Any

ParameterSetByAtomIndex = Dict[Tuple[int, ...], Dict[str, Any]]
ParameterSetByNode = Dict[Tuple[Tuple[str, int], ...], Dict[str, Any]]
ForceFieldParametersByAtomIndex = Dict[str, ParameterSetByAtomIndex]
ForceFieldParametersByNode = Dict[str, ParameterSetByNode]
