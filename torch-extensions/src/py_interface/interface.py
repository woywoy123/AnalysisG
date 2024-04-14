import torch
from torch import Tensor
from typing import Dict, List, Tuple

def pyc_path(): return "/".join(__file__.split("/")[:-1]) + "/libpyc_cuda.so"

@torch.jit.script
def cuPxPyPzE(pmu: Tensor) -> Tensor: return torch.ops.pyc_cuda.transform_combined_PxPyPzE(pmu)

@torch.jit.script
def cuPx(pt: Tensor, phi: Tensor) -> Tensor: return torch.ops.pyc_cuda.transform_separate_Px(pt, phi)

@torch.jit.script
def cuPy(pt: Tensor, phi: Tensor) -> Tensor: return torch.ops.pyc_cuda.transform_separate_Py(pt, phi)

@torch.jit.script
def cuPz(pt: Tensor, eta: Tensor) -> Tensor: return torch.ops.pyc_cuda.transform_separate_Pz(pt, eta)

@torch.jit.script
def cuNuNuCombinatorial(edge_index: Tensor, batch: Tensor, pmc: Tensor, pid: Tensor, met_xy: Tensor, gev: bool = False) -> Dict[str, Tensor]:
    output: List[Tensor] = torch.ops.pyc_cuda.combinatorial(edge_index, batch, pmc, pid, met_xy, gev)
    return {"nu_1f" : output[0], "nu_2f" : output[1], "ms_1f" : output[2], "ms_2f" : output[3], "combi" : output[4], "min" : output[5]}

@torch.jit.script
def cuUniqueAggregation(trk_i: Tensor, pmc: Tensor) -> Tuple[Tensor, Tensor]:
    return torch.ops.pyc_cuda.graph_base_combined_unique_aggregation(trk_i, pmc)

@torch.jit.script
def cuMass(pmc: Tensor) -> Tensor: return torch.ops.pyc_cuda.physics_combined_cartesian_M(pmc)

@torch.jit.script
def cuCdeltaR(pmc1: Tensor, pmc2: Tensor) -> Tensor:
    return torch.ops.pyc_cuda.physics_combined_cartesian_DeltaR(pmc1, pmc2)

@torch.jit.script
def cuEdgeAggregation(edge_index: Tensor, message: Tensor, trk: Tensor, cls: int) -> Dict[str, Tensor]:
    output: List[Tensor] = torch.ops.pyc_cuda.graph_base_combined_edge_aggregation(edge_index, message, trk)[cls]
    return {"clusters" : output[0], "unique_sum" : output[1], "reverse_clusters" : output[2], "node_sum" : output[3]}
