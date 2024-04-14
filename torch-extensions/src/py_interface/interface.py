import torch
from torch import Tensor
from typing import Dict, List, Tuple

def pyc_cuda_(): return "/".join(__file__.split("/")[:-1]) + "/libpyc_cuda.so"
CUDA_PATH = pyc_cuda_()

def pyc_tensor_(): return "/".join(__file__.split("/")[:-1]) + "/libpyc_tensor.so"
TENSOR_PATH = pyc_tensor_()

@torch.jit.script
class pyc_cuda:
    torch.ops.load_library(pyc_cuda_())

    @torch.jit.script
    class separate:

        @torch.jit.script
        class transform:

            @torch.jit.script
            def Px(pt: Tensor, phi: Tensor) -> Tensor:
                return torch.ops.pyc_cuda.transform_separate_Px(pt, phi)

            @torch.jit.script
            def Py(pt: Tensor, phi: Tensor) -> Tensor:
                return torch.ops.pyc_cuda.transform_separate_Py(pt, phi)

            @torch.jit.script
            def Pz(pt: Tensor, eta: Tensor) -> Tensor:
                return torch.ops.pyc_cuda.transform_separate_Pz(pt, eta)

            @torch.jit.script
            def PxPyPz(pt: Tensor, eta: Tensor, phi: Tensor) -> Tensor:
                return torch.ops.pyc_cuda.transform_separate_PxPyPz(pt, eta, phi)

            @torch.jit.script
            def PxPyPzE(pt: Tensor, eta: Tensor, phi: Tensor, e: Tensor) -> Tensor:
                return torch.ops.pyc_cuda.transform_separate_PxPyPzE(pt, eta, phi, e)

            @torch.jit.script
            def Pt(px: Tensor, py: Tensor) -> Tensor:
                return torch.ops.pyc_cuda.transform_separate_Pt(px, py)

            @torch.jit.script
            def Phi(px: Tensor, py: Tensor) -> Tensor:
                return torch.ops.pyc_cuda.transform_separate_Phi(px, py)

            @torch.jit.script
            def Eta(px: Tensor, py: Tensor, pz: Tensor) -> Tensor:
                return torch.ops.pyc_cuda.transform_separate_Eta(px, py, pz)

            @torch.jit.script
            def PtEtaPhi(px: Tensor, py: Tensor, pz: Tensor) -> Tensor:
                return torch.ops.pyc_cuda.transform_separate_PtEtaPhi(px, py, pz)

            @torch.jit.script
            def PtEtaPhiE(px: Tensor, py: Tensor, pz: Tensor, e: Tensor) -> Tensor:
                return torch.ops.pyc_cuda.transform_separate_PtEtaPhiE(px, py, pz, e)

        @torch.jit.script
        class physics:

            @torch.jit.script
            class cartesian:

                @torch.jit.script
                def P2(px: Tensor, py: Tensor, pz: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_separate_cartesian_P2(px, py, pz)

                @torch.jit.script
                def P(px: Tensor, py: Tensor, pz: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_separate_cartesian_P(px, py, pz)

                @torch.jit.script
                def Beta2(px: Tensor, py: Tensor, pz: Tensor, e: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_separate_cartesian_Beta2(px, py, pz, e)

                @torch.jit.script
                def Beta(px: Tensor, py: Tensor, pz: Tensor, e: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_separate_cartesian_Beta(px, py, pz, e)

                @torch.jit.script
                def M2(px: Tensor, py: Tensor, pz: Tensor, e: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_separate_cartesian_M2(px, py, pz, e)

                @torch.jit.script
                def M(px: Tensor, py: Tensor, pz: Tensor, e: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_separate_cartesian_M(px, py, pz, e)

                @torch.jit.script
                def Mt2(pz: Tensor, e: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_separate_cartesian_Mt2(pz, e)

                @torch.jit.script
                def Mt(pz: Tensor, e: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_separate_cartesian_Mt(pz, e)

                @torch.jit.script
                def Theta(px: Tensor, py: Tensor, pz: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_separate_cartesian_Theta(px, py, pz)

                @torch.jit.script
                def DeltaR(px1: Tensor, px2: Tensor, py1: Tensor, py2: Tensor, pz1: Tensor, pz2: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_separate_cartesian_DeltaR(px1, px2, py1, py2, pz1, pz2)

            @torch.jit.script
            class polar:

                @torch.jit.script
                def P2(pt: Tensor, eta: Tensor, phi: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_separate_polar_P2(pt, eta, phi)

                @torch.jit.script
                def P(pt: Tensor, eta: Tensor, phi: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_separate_polar_P(pt, eta, phi)

                @torch.jit.script
                def Beta2(pt: Tensor, eta: Tensor, phi: Tensor, e: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_separate_polar_Beta2(pt, eta, phi, e)

                @torch.jit.script
                def Beta(pt: Tensor, eta: Tensor, phi: Tensor, e: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_separate_polar_Beta(pt, eta, phi, e)

                @torch.jit.script
                def M2(pt: Tensor, eta: Tensor, phi: Tensor, e: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_separate_polar_M2(pt, eta, phi, e)

                @torch.jit.script
                def M(pt: Tensor, eta: Tensor, phi: Tensor, e: Tensor) -> Tensor: 
                    return torch.ops.pyc_cuda.physics_separate_polar_M(pt, eta, phi, e)

                @torch.jit.script
                def Mt2(pt: Tensor, eta: Tensor, e: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_separate_polar_Mt2(pt, eta, e)

                @torch.jit.script
                def Mt(pt: Tensor, eta: Tensor, e: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_separate_polar_Mt(pt, eta, e)

                @torch.jit.script
                def Theta(pt: Tensor, eta: Tensor, phi: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_separate_polar_Theta(pt, eta, phi)

                @torch.jit.script
                def DeltaR(eta1: Tensor, eta2: Tensor, phi1: Tensor, phi2: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_separate_polar_DeltaR(eta1, eta2, phi1, phi2)

        @torch.jit.script
        class nusol:

            @torch.jit.script
            class polar:

                @torch.jit.script
                def Nu(
                        pt_b: Tensor, eta_b: Tensor, phi_b: Tensor, e_b: Tensor,
                        pt_mu: Tensor, eta_mu: Tensor, phi_mu: Tensor, e_mu: Tensor,
                        met: Tensor, phi: Tensor, masses: Tensor, sigma: Tensor, null: float = 1e-10
                    ) -> Dict[str, Tensor]:

                    result: List[Tensor] = torch.ops.pyc_cuda.nusol_separate_polar_Nu(
                            pt_b, eta_b, phi_b, e_b, pt_mu, eta_mu, phi_mu, e_mu, met, phi, masses, sigma, null
                    )
                    return {"NuVec" : result[0], "chi2" : result[1]};

                @torch.jit.script
                def NuNu(
                        pt_b1: Tensor, eta_b1: Tensor, phi_b1: Tensor, e_b1: Tensor,
                        pt_b2: Tensor, eta_b2: Tensor, phi_b2: Tensor, e_b2: Tensor,
                        pt_mu1: Tensor, eta_mu1: Tensor, phi_mu1: Tensor, e_mu1: Tensor,
                        pt_mu2: Tensor, eta_mu2: Tensor, phi_mu2: Tensor, e_mu2: Tensor,
                        met: Tensor, phi: Tensor, masses: Tensor, null: float = 1e-10
                    ) -> Dict[str, Tensor]:
                        out: List[Tensor] = torch.ops.pyc_cuda.nusol_separate_polar_NuNu(
                                pt_b1,  eta_b1 , phi_b1 , e_b1 , pt_b2 , eta_b2 , phi_b2 , e_b2 ,
                                pt_mu1, eta_mu1, phi_mu1, e_mu1, pt_mu2, eta_mu2, phi_mu2, e_mu2,
                                met, phi, masses, null
                        )
                        return {
                                "NuVec_1": out[0] , "NuVec_2": out[1], "Diagonal": out[2], "n_": out[3],
                                "H_perp_1" : out[4], "H_perp_2" : out[5], "NoSols" : out[6]
                        }
            @torch.jit.script
            class cartesian:

                @torch.jit.script
                def Nu(
                        px_b: Tensor, py_b: Tensor, pz_b: Tensor, e_b: Tensor,
                        px_mu: Tensor, py_mu: Tensor, pz_mu: Tensor, e_mu: Tensor,
                        metx: Tensor, mety: Tensor, masses: Tensor, sigma: Tensor,
                        null: float = 1e-10
                    ) -> Dict[str, Tensor]:

                    result: List[Tensor] = torch.ops.pyc_cuda.nusol_separate_cartesian_Nu(
                            px_b, py_b, pz_b, e_b, px_mu, py_mu, pz_mu, e_mu, metx, mety, masses, sigma, null
                    )
                    return {"NuVec" : result[0], "chi2" : result[1]};

                @torch.jit.script
                def NuNu(
                        px_b1: Tensor, py_b1: Tensor, pz_b1: Tensor, e_b1: Tensor,
                        px_b2: Tensor, py_b2: Tensor, pz_b2: Tensor, e_b2: Tensor,
                        px_mu1: Tensor, py_mu1: Tensor, pz_mu1: Tensor, e_mu1: Tensor,
                        px_mu2: Tensor, py_mu2: Tensor, pz_mu2: Tensor, e_mu2: Tensor,
                        metx: Tensor, mety: Tensor, masses: Tensor,
                        null: float = 1e-10
                    ) -> Dict[str, Tensor]:
                        out: List[Tensor] = torch.ops.pyc_cuda.nusol_separate_cartesian_NuNu(
                                px_b1 , py_b1 , pz_b1 , e_b1 , px_b2 , py_b2 , pz_b2 , e_b2 ,
                                px_mu1, py_mu1, pz_mu1, e_mu1, px_mu2, py_mu2, pz_mu2, e_mu2,
                                metx, mety, masses, null
                        )
                        return {
                                "NuVec_1": out[0] , "NuVec_2": out[1], "Diagonal": out[2], "n_": out[3],
                                "H_perp_1" : out[4], "H_perp_2" : out[5], "NoSols" : out[6]
                        }


    @torch.jit.script
    class combined:

        @torch.jit.script
        class transform:

            @torch.jit.script
            def Px(pmu: Tensor) -> Tensor:
                return torch.ops.pyc_cuda.transform_combined_Px(pmu)

            @torch.jit.script
            def Py(pmu: Tensor) -> Tensor:
                return torch.ops.pyc_cuda.transform_combined_Py(pmu)

            @torch.jit.script
            def Pz(pmu: Tensor) -> Tensor:
                return torch.ops.pyc_cuda.transform_combined_Pz(pmu)

            @torch.jit.script
            def Pt(pmu: Tensor) -> Tensor:
                return torch.ops.pyc_cuda.transform_combined_Pt(pmu)

            @torch.jit.script
            def PxPyPz(pmu: Tensor) -> Tensor:
                return torch.ops.pyc_cuda.transform_combined_PxPyPz(pmu)

            @torch.jit.script
            def PxPyPzE(pmu: Tensor) -> Tensor:
                return torch.ops.pyc_cuda.transform_combined_PxPyPzE(pmu)

            @torch.jit.script
            def Pt(pmc: Tensor) -> Tensor:
                return torch.ops.pyc_cuda.transform_combined_Pt(pmc)

            @torch.jit.script
            def Phi(pmc: Tensor) -> Tensor:
                return torch.ops.pyc_cuda.transform_combined_Phi(pmc)

            @torch.jit.script
            def Eta(pmc: Tensor) -> Tensor:
                return torch.ops.pyc_cuda.transform_combined_Eta(pmc)

            @torch.jit.script
            def PtEtaPhi(pmc: Tensor) -> Tensor:
                return torch.ops.pyc_cuda.transform_combined_PtEtaPhi(pmc)

            @torch.jit.script
            def PtEtaPhiE(pmc: Tensor) -> Tensor:
                return torch.ops.pyc_cuda.transform_combined_PtEtaPhiE(pmc)

        @torch.jit.script
        class physics:

            @torch.jit.script
            class cartesian:

                @torch.jit.script
                def P2(pmc: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_combined_cartesian_P2(pmc)

                @torch.jit.script
                def P(pmc: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_combined_cartesian_P(pmc)

                @torch.jit.script
                def Beta2(pmc: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_combined_cartesian_Beta2(pmc)

                @torch.jit.script
                def Beta(pmc: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_combined_cartesian_Beta(pmc)

                @torch.jit.script
                def M2(pmc: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_combined_cartesian_M2(pmc)

                @torch.jit.script
                def M(pmc: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_combined_cartesian_M(pmc)

                @torch.jit.script
                def Mt2(pmc: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_combined_cartesian_Mt2(pmc)

                @torch.jit.script
                def Mt(pmc: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_combined_cartesian_Mt(pmc)

                @torch.jit.script
                def Theta(pmc: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_combined_cartesian_Theta(pmc)

                @torch.jit.script
                def DeltaR(pmc1: Tensor, pmc2: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_combined_cartesian_DeltaR(pmc1, pmc2)

            @torch.jit.script
            class polar:

                @torch.jit.script
                def P2(pmu: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_combined_polar_P2(pmu)

                @torch.jit.script
                def P(pmu: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_combined_polar_P(pmu)

                @torch.jit.script
                def Beta2(pmu: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_combined_polar_Beta2(pmu)

                @torch.jit.script
                def Beta(pmu: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_combined_polar_Beta(pmu)

                @torch.jit.script
                def M2(pmu: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_combined_polar_M2(pmu)

                @torch.jit.script
                def M(pmu: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_combined_polar_M(pmu)

                @torch.jit.script
                def Mt2(pmu: Tensor) -> Tensor:
                   return torch.ops.pyc_cuda.physics_combined_polar_Mt2(pmu)

                @torch.jit.script
                def Mt(pmu: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_combined_polar_Mt(pmu)

                @torch.jit.script
                def Theta(pmu: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_combined_polar_Theta(pmu)

                @torch.jit.script
                def DeltaR(pmu1: Tensor, pmu2: Tensor) -> Tensor:
                    return torch.ops.pyc_cuda.physics_combined_polar_DeltaR(pmu1, pmu2)


        @torch.jit.script
        class nusol:

            @torch.jit.script
            class polar:

                @torch.jit.script
                def Nu(pmu_b: Tensor, pmu_mu: Tensor, met_phi: Tensor, masses: Tensor, sigma: Tensor, null: float = 1e-10) -> Dict[str, Tensor]:
                    out: List[Tensor] = torch.ops.pyc_cuda.nusol_combined_polar_Nu(pmu_b, pmu_mu, met_phi, masses, sigma, null)
                    return {"NuVec" : out[0], "chi2" : out[1]};

                @torch.jit.script
                def NuNu(pmu_b1: Tensor, pmu_b2: Tensor, pmu_mu1: Tensor, pmu_mu2: Tensor, met_phi: Tensor, masses: Tensor, null: float = 1e-10) -> Dict[str, Tensor]:
                    out: List[Tensor] = torch.ops.pyc_cuda.nusol_combined_polar_NuNu(pmu_b1, pmu_b2, pmu_mu1, pmu_mu2, met_phi, masses, null)
                    return {"NuVec_1": out[0] , "NuVec_2": out[1], "Diagonal": out[2], "n_": out[3], "H_perp_1" : out[4], "H_perp_2" : out[5], "NoSols" : out[6]}


            @torch.jit.script
            class cartesian:

                @torch.jit.script
                def Nu(pmc_b: Tensor, pmc_mu: Tensor, met_xy: Tensor, masses: Tensor, sigma: Tensor, null: float = 1e-10) -> Dict[str, Tensor]:
                    result: List[Tensor] = torch.ops.pyc_cuda.nusol_combined_cartesian_Nu(pmc_b, pmc_mu, met_xy, masses, sigma, null)
                    return {"NuVec" : result[0], "chi2" : result[1]}

                @torch.jit.script
                def NuNu(pmc_b1: Tensor, pmc_b2: Tensor, pmc_mu1: Tensor, pmc_mu2, met_xy: Tensor, masses: Tensor, null: float = 1e-10) -> Dict[str, Tensor]:
                        out: List[Tensor] = torch.ops.pyc_cuda.nusol_combined_cartesian_NuNu(pmc_b1, pmc_b2, pmc_mu1, pmc_mu2, met_xy, masses, null)
                        return {
                                "NuVec_1": out[0] , "NuVec_2": out[1], "Diagonal": out[2], "n_": out[3],
                                "H_perp_1" : out[4], "H_perp_2" : out[5], "NoSols" : out[6]
                        }

    @torch.jit.script
    class nusol:

        @torch.jit.script
        def BaseMatrix(pmc_b: Tensor, pmc_mu: Tensor, masses: Tensor) -> Tensor:
            return torch.ops.pyc_cuda.nusol_BaseMatrix(pmc_b, pmc_mu, masses)

        @torch.jit.script
        def Intersection(A: Tensor, B: Tensor, null: float = 1e-10) -> Tuple[Tensor, Tensor]:
            return torch.ops.pyc_cuda.nusol_Intersection(A, B, null)

        @torch.jit.script
        def Nu(pmc_b: Tensor, pmc_mu: Tensor, met_xy: Tensor, masses: Tensor, sigma: Tensor, null: float = 1e-10) -> List[Tensor]:
            return torch.ops.pyc_cuda.nusol_Nu(pmc_b, pmc_mu, met_xy, masses, sigma, null)

        @torch.jit.script
        def NuNu(pmc_b: Tensor, pmc_mu: Tensor, met_xy: Tensor, masses: Tensor, sigma: Tensor, null: float = 1e-10) -> List[Tensor]:
            return torch.ops.pyc_cuda.nusol_Nu(pmc_b, pmc_mu, met_xy, masses, sigma, null)

        @torch.jit.script
        def NuNu(pmc_b1: Tensor, pmc_b2: Tensor, pmc_l1: Tensor, pmc_l2: Tensor, met_xy: Tensor, masses: Tensor, null: float = 1e-10) -> List[Tensor]:
            return torch.ops.pyc_cuda.nusol_NuNu(pmc_b1, pmc_b2, pmc_l1, pmc_l2, met_xy, masses, null)

        @torch.jit.script
        def combinatorial(edge_index: Tensor, batch: Tensor, pmc: Tensor, pid: Tensor, met_xy: Tensor, gev: bool = False) -> Dict[str, Tensor]:
            result: List[Tensor] = torch.ops.pyc_cuda.combinatorial(edge_index, batch, pmc, pid, met_xy, gev)
            comb: Tensor = result[4].sum(-1) > 0
            l1: Tensor = result[4][comb, 2].to(dtype = torch.int64)
            l2: Tensor = result[4][comb, 3].to(dtype = torch.int64)
            pmc = pmc.clone()
            pmc[l1] += result[0][comb]
            pmc[l2] += result[1][comb]
            return {"nu_1f": result[0], "nu_2f": result[1], "masses_nu1": result[2], "masses_nu2": result[3], "combination": result[4], "minimum" : result[5], "pmc" : pmc}

    @torch.jit.script
    class graph:

        @torch.jit.script
        def edge_aggregation(edge_index: Tensor, prediction: Tensor, node_feature: Tensor) -> Dict[int, Dict[str, Tensor]]:
            res: List[List[Tensor]] = torch.ops.pyc_cuda.graph_base_combined_edge_aggregation(edge_index, prediction, node_feature)
            idx: int = len(res)
            output: Dict[int, Dict[str, Tensor]] = {}
            for ix in range(idx): output[ix] = {"clusters": res[ix][0], "unique_sum": res[ix][1], "reverse_clusters": res[ix][2], "node_sum": res[ix][3]}
            return output

        @torch.jit.script
        def node_aggregation(edge_index: Tensor, prediction: Tensor, node_feature: Tensor) -> Dict[int, Dict[str, Tensor]]:
            res: List[List[Tensor]] = torch.ops.pyc_cuda.graph_base_combined_node_aggregation(edge_index, prediction, node_feature)
            idx: int = len(res)
            output: Dict[int, Dict[str, Tensor]] = {}
            for ix in range(idx): output[ix] = {"clusters": res[ix][0], "unique_sum": res[ix][1], "reverse_clusters": res[ix][2], "node_sum": res[ix][3]}
            return output

        @torch.jit.script
        def unique_aggregation(cluster_map: Tensor, features: Tensor) -> Tuple[Tensor, Tensor]:
            return torch.ops.pyc_cuda.graph_base_combined_unique_aggregation(cluster_map, features)

    @torch.jit.script
    class operators:

        @torch.jit.script
        def Dot(v1: Tensor, v2: Tensor) -> Tensor:
            return torch.ops.pyc_cuda.operators_Dot(v1, v2)

        @torch.jit.script
        def Mul(v1: Tensor, v2: Tensor) -> Tensor:
            return torch.ops.pyc_cuda.operators_Mul(v1, v2)

        @torch.jit.script
        def CosTheta(v1: Tensor, v2: Tensor) -> Tensor:
            return torch.ops.pyc_cuda.operators_CosTheta(v1, v2)

        @torch.jit.script
        def SinTheta(v1: Tensor, v2: Tensor) -> Tensor:
            return torch.ops.pyc_cuda.operators_SinTheta(v1, v2)

        @torch.jit.script
        def Rx(angle: Tensor) -> Tensor:
            return torch.ops.pyc_cuda.operators_Rx(angle)

        @torch.jit.script
        def Ry(angle: Tensor) -> Tensor:
            return torch.ops.pyc_cuda.operators_Ry(angle)

        @torch.jit.script
        def Rz(angle: Tensor) -> Tensor:
            return torch.ops.pyc_cuda.operators_Rz(angle)

        @torch.jit.script
        def CoFactors(matrix: Tensor) -> Tensor:
            return torch.ops.pyc_cuda.operators_CoFactors(matrix)

        @torch.jit.script
        def Determinant(matrix: Tensor) -> Tensor:
            return torch.ops.pyc_cuda.operators_Determinant(matrix)

        @torch.jit.script
        def Inverse(matrix: Tensor) -> Tensor:
            return torch.ops.pyc_cuda.operators_Inverse(matrix)

        @torch.jit.script
        def Cross(mat1: Tensor, mat2: Tensor) -> Tensor:
            return torch.ops.pyc_cuda.operators_Cross(mat1, mat2)


@torch.jit.script
class pyc_tensor:
    torch.ops.load_library(pyc_tensor_())

    @torch.jit.script
    class separate:

        @torch.jit.script
        class transform:

            @torch.jit.script
            def Px(pt: Tensor, phi: Tensor) -> Tensor:
                return torch.ops.pyc_tensor.transform_separate_Px(pt, phi)

            @torch.jit.script
            def Py(pt: Tensor, phi: Tensor) -> Tensor:
                return torch.ops.pyc_tensor.transform_separate_Py(pt, phi)

            @torch.jit.script
            def Pz(pt: Tensor, eta: Tensor) -> Tensor:
                return torch.ops.pyc_tensor.transform_separate_Pz(pt, eta)

            @torch.jit.script
            def PxPyPz(pt: Tensor, eta: Tensor, phi: Tensor) -> Tensor:
                return torch.ops.pyc_tensor.transform_separate_PxPyPz(pt, eta, phi)

            @torch.jit.script
            def PxPyPzE(pt: Tensor, eta: Tensor, phi: Tensor, e: Tensor) -> Tensor:
                return torch.ops.pyc_tensor.transform_separate_PxPyPzE(pt, eta, phi, e)

            @torch.jit.script
            def Pt(px: Tensor, py: Tensor) -> Tensor:
                return torch.ops.pyc_tensor.transform_separate_Pt(px, py)

            @torch.jit.script
            def Phi(px: Tensor, py: Tensor) -> Tensor:
                return torch.ops.pyc_tensor.transform_separate_Phi(px, py)

            @torch.jit.script
            def Eta(px: Tensor, py: Tensor, pz: Tensor) -> Tensor:
                return torch.ops.pyc_tensor.transform_separate_Eta(px, py, pz)

            @torch.jit.script
            def PtEtaPhi(px: Tensor, py: Tensor, pz: Tensor) -> Tensor:
                return torch.ops.pyc_tensor.transform_separate_PtEtaPhi(px, py, pz)

            @torch.jit.script
            def PtEtaPhiE(px: Tensor, py: Tensor, pz: Tensor, e: Tensor) -> Tensor:
                return torch.ops.pyc_tensor.transform_separate_PtEtaPhiE(px, py, pz, e)

        @torch.jit.script
        class physics:

            @torch.jit.script
            class cartesian:

                @torch.jit.script
                def P2(px: Tensor, py: Tensor, pz: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_separate_cartesian_P2(px, py, pz)

                @torch.jit.script
                def P(px: Tensor, py: Tensor, pz: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_separate_cartesian_P(px, py, pz)

                @torch.jit.script
                def Beta2(px: Tensor, py: Tensor, pz: Tensor, e: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_separate_cartesian_Beta2(px, py, pz, e)

                @torch.jit.script
                def Beta(px: Tensor, py: Tensor, pz: Tensor, e: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_separate_cartesian_Beta(px, py, pz, e)

                @torch.jit.script
                def M2(px: Tensor, py: Tensor, pz: Tensor, e: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_separate_cartesian_M2(px, py, pz, e)

                @torch.jit.script
                def M(px: Tensor, py: Tensor, pz: Tensor, e: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_separate_cartesian_M(px, py, pz, e)

                @torch.jit.script
                def Mt2(pz: Tensor, e: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_separate_cartesian_Mt2(pz, e)

                @torch.jit.script
                def Mt(pz: Tensor, e: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_separate_cartesian_Mt(pz, e)

                @torch.jit.script
                def Theta(px: Tensor, py: Tensor, pz: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_separate_cartesian_Theta(px, py, pz)

                @torch.jit.script
                def DeltaR(px1: Tensor, px2: Tensor, py1: Tensor, py2: Tensor, pz1: Tensor, pz2: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_separate_cartesian_DeltaR(px1, px2, py1, py2, pz1, pz2)

            @torch.jit.script
            class polar:

                @torch.jit.script
                def P2(pt: Tensor, eta: Tensor, phi: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_separate_polar_P2(pt, eta, phi)

                @torch.jit.script
                def P(pt: Tensor, eta: Tensor, phi: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_separate_polar_P(pt, eta, phi)

                @torch.jit.script
                def Beta2(pt: Tensor, eta: Tensor, phi: Tensor, e: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_separate_polar_Beta2(pt, eta, phi, e)

                @torch.jit.script
                def Beta(pt: Tensor, eta: Tensor, phi: Tensor, e: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_separate_polar_Beta(pt, eta, phi, e)

                @torch.jit.script
                def M2(pt: Tensor, eta: Tensor, phi: Tensor, e: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_separate_polar_M2(pt, eta, phi, e)

                @torch.jit.script
                def M(pt: Tensor, eta: Tensor, phi: Tensor, e: Tensor) -> Tensor: 
                    return torch.ops.pyc_tensor.physics_separate_polar_M(pt, eta, phi, e)

                @torch.jit.script
                def Mt2(pt: Tensor, eta: Tensor, e: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_separate_polar_Mt2(pt, eta, e)

                @torch.jit.script
                def Mt(pt: Tensor, eta: Tensor, e: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_separate_polar_Mt(pt, eta, e)

                @torch.jit.script
                def Theta(pt: Tensor, eta: Tensor, phi: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_separate_polar_Theta(pt, eta, phi)

                @torch.jit.script
                def DeltaR(eta1: Tensor, eta2: Tensor, phi1: Tensor, phi2: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_separate_polar_DeltaR(eta1, eta2, phi1, phi2)

        @torch.jit.script
        class nusol:

            @torch.jit.script
            class polar:

                @torch.jit.script
                def Nu(
                        pt_b: Tensor, eta_b: Tensor, phi_b: Tensor, e_b: Tensor,
                        pt_mu: Tensor, eta_mu: Tensor, phi_mu: Tensor, e_mu: Tensor,
                        met: Tensor, phi: Tensor, masses: Tensor, sigma: Tensor, null: float = 1e-10
                    ) -> Dict[str, Tensor]:

                    result: List[Tensor] = torch.ops.pyc_tensor.nusol_separate_polar_Nu(
                            pt_b, eta_b, phi_b, e_b, pt_mu, eta_mu, phi_mu, e_mu, met, phi, masses, sigma, null
                    )
                    return {"NuVec" : result[0], "chi2" : result[1]};

                @torch.jit.script
                def NuNu(
                        pt_b1: Tensor, eta_b1: Tensor, phi_b1: Tensor, e_b1: Tensor,
                        pt_b2: Tensor, eta_b2: Tensor, phi_b2: Tensor, e_b2: Tensor,
                        pt_mu1: Tensor, eta_mu1: Tensor, phi_mu1: Tensor, e_mu1: Tensor,
                        pt_mu2: Tensor, eta_mu2: Tensor, phi_mu2: Tensor, e_mu2: Tensor,
                        met: Tensor, phi: Tensor, masses: Tensor, null: float = 1e-10
                    ) -> Dict[str, Tensor]:
                        out: List[Tensor] = torch.ops.pyc_tensor.nusol_separate_polar_NuNu(
                                pt_b1,  eta_b1 , phi_b1 , e_b1 , pt_b2 , eta_b2 , phi_b2 , e_b2 ,
                                pt_mu1, eta_mu1, phi_mu1, e_mu1, pt_mu2, eta_mu2, phi_mu2, e_mu2,
                                met, phi, masses, null
                        )
                        return {
                                "NuVec_1": out[0] , "NuVec_2": out[1], "Diagonal": out[2], "n_": out[3],
                                "H_perp_1" : out[4], "H_perp_2" : out[5], "NoSols" : out[6]
                        }
            @torch.jit.script
            class cartesian:

                @torch.jit.script
                def Nu(
                        px_b: Tensor, py_b: Tensor, pz_b: Tensor, e_b: Tensor,
                        px_mu: Tensor, py_mu: Tensor, pz_mu: Tensor, e_mu: Tensor,
                        metx: Tensor, mety: Tensor, masses: Tensor, sigma: Tensor,
                        null: float = 1e-10
                    ) -> Dict[str, Tensor]:

                    result: List[Tensor] = torch.ops.pyc_tensor.nusol_separate_cartesian_Nu(
                            px_b, py_b, pz_b, e_b, px_mu, py_mu, pz_mu, e_mu, metx, mety, masses, sigma, null
                    )
                    return {"NuVec" : result[0], "chi2" : result[1]};

                @torch.jit.script
                def NuNu(
                        px_b1: Tensor, py_b1: Tensor, pz_b1: Tensor, e_b1: Tensor,
                        px_b2: Tensor, py_b2: Tensor, pz_b2: Tensor, e_b2: Tensor,
                        px_mu1: Tensor, py_mu1: Tensor, pz_mu1: Tensor, e_mu1: Tensor,
                        px_mu2: Tensor, py_mu2: Tensor, pz_mu2: Tensor, e_mu2: Tensor,
                        metx: Tensor, mety: Tensor, masses: Tensor,
                        null: float = 1e-10
                    ) -> Dict[str, Tensor]:
                        out: List[Tensor] = torch.ops.pyc_tensor.nusol_separate_cartesian_NuNu(
                                px_b1 , py_b1 , pz_b1 , e_b1 , px_b2 , py_b2 , pz_b2 , e_b2 ,
                                px_mu1, py_mu1, pz_mu1, e_mu1, px_mu2, py_mu2, pz_mu2, e_mu2,
                                metx, mety, masses, null
                        )
                        return {
                                "NuVec_1": out[0] , "NuVec_2": out[1], "Diagonal": out[2], "n_": out[3],
                                "H_perp_1" : out[4], "H_perp_2" : out[5], "NoSols" : out[6]
                        }


    @torch.jit.script
    class combined:

        @torch.jit.script
        class transform:

            @torch.jit.script
            def Px(pmu: Tensor) -> Tensor:
                return torch.ops.pyc_tensor.transform_combined_Px(pmu)

            @torch.jit.script
            def Py(pmu: Tensor) -> Tensor:
                return torch.ops.pyc_tensor.transform_combined_Py(pmu)

            @torch.jit.script
            def Pz(pmu: Tensor) -> Tensor:
                return torch.ops.pyc_tensor.transform_combined_Pz(pmu)

            @torch.jit.script
            def Pt(pmu: Tensor) -> Tensor:
                return torch.ops.pyc_tensor.transform_combined_Pt(pmu)

            @torch.jit.script
            def PxPyPz(pmu: Tensor) -> Tensor:
                return torch.ops.pyc_tensor.transform_combined_PxPyPz(pmu)

            @torch.jit.script
            def PxPyPzE(pmu: Tensor) -> Tensor:
                return torch.ops.pyc_tensor.transform_combined_PxPyPzE(pmu)

            @torch.jit.script
            def Pt(pmc: Tensor) -> Tensor:
                return torch.ops.pyc_tensor.transform_combined_Pt(pmc)

            @torch.jit.script
            def Phi(pmc: Tensor) -> Tensor:
                return torch.ops.pyc_tensor.transform_combined_Phi(pmc)

            @torch.jit.script
            def Eta(pmc: Tensor) -> Tensor:
                return torch.ops.pyc_tensor.transform_combined_Eta(pmc)

            @torch.jit.script
            def PtEtaPhi(pmc: Tensor) -> Tensor:
                return torch.ops.pyc_tensor.transform_combined_PtEtaPhi(pmc)

            @torch.jit.script
            def PtEtaPhiE(pmc: Tensor) -> Tensor:
                return torch.ops.pyc_tensor.transform_combined_PtEtaPhiE(pmc)

        @torch.jit.script
        class physics:

            @torch.jit.script
            class cartesian:

                @torch.jit.script
                def P2(pmc: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_combined_cartesian_P2(pmc)

                @torch.jit.script
                def P(pmc: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_combined_cartesian_P(pmc)

                @torch.jit.script
                def Beta2(pmc: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_combined_cartesian_Beta2(pmc)

                @torch.jit.script
                def Beta(pmc: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_combined_cartesian_Beta(pmc)

                @torch.jit.script
                def M2(pmc: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_combined_cartesian_M2(pmc)

                @torch.jit.script
                def M(pmc: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_combined_cartesian_M(pmc)

                @torch.jit.script
                def Mt2(pmc: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_combined_cartesian_Mt2(pmc)

                @torch.jit.script
                def Mt(pmc: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_combined_cartesian_Mt(pmc)

                @torch.jit.script
                def Theta(pmc: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_combined_cartesian_Theta(pmc)

                @torch.jit.script
                def DeltaR(pmc1: Tensor, pmc2: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_combined_cartesian_DeltaR(pmc1, pmc2)

            @torch.jit.script
            class polar:

                @torch.jit.script
                def P2(pmu: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_combined_polar_P2(pmu)

                @torch.jit.script
                def P(pmu: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_combined_polar_P(pmu)

                @torch.jit.script
                def Beta2(pmu: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_combined_polar_Beta2(pmu)

                @torch.jit.script
                def Beta(pmu: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_combined_polar_Beta(pmu)

                @torch.jit.script
                def M2(pmu: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_combined_polar_M2(pmu)

                @torch.jit.script
                def M(pmu: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_combined_polar_M(pmu)

                @torch.jit.script
                def Mt2(pmu: Tensor) -> Tensor:
                   return torch.ops.pyc_tensor.physics_combined_polar_Mt2(pmu)

                @torch.jit.script
                def Mt(pmu: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_combined_polar_Mt(pmu)

                @torch.jit.script
                def Theta(pmu: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_combined_polar_Theta(pmu)

                @torch.jit.script
                def DeltaR(pmu1: Tensor, pmu2: Tensor) -> Tensor:
                    return torch.ops.pyc_tensor.physics_combined_polar_DeltaR(pmu1, pmu2)


        @torch.jit.script
        class nusol:

            @torch.jit.script
            class polar:

                @torch.jit.script
                def Nu(pmu_b: Tensor, pmu_mu: Tensor, met_phi: Tensor, masses: Tensor, sigma: Tensor, null: float = 1e-10) -> Dict[str, Tensor]:
                    out: List[Tensor] = torch.ops.pyc_tensor.nusol_combined_polar_Nu(pmu_b, pmu_mu, met_phi, masses, sigma, null)
                    return {"NuVec" : out[0], "chi2" : out[1]};

                @torch.jit.script
                def NuNu(pmu_b1: Tensor, pmu_b2: Tensor, pmu_mu1: Tensor, pmu_mu2: Tensor, met_phi: Tensor, masses: Tensor, null: float = 1e-10) -> Dict[str, Tensor]:
                    out: List[Tensor] = torch.ops.pyc_tensor.nusol_combined_polar_NuNu(pmu_b1, pmu_b2, pmu_mu1, pmu_mu2, met_phi, masses, null)
                    return {"NuVec_1": out[0] , "NuVec_2": out[1], "Diagonal": out[2], "n_": out[3], "H_perp_1" : out[4], "H_perp_2" : out[5], "NoSols" : out[6]}


            @torch.jit.script
            class cartesian:

                @torch.jit.script
                def Nu(pmc_b: Tensor, pmc_mu: Tensor, met_xy: Tensor, masses: Tensor, sigma: Tensor, null: float = 1e-10) -> Dict[str, Tensor]:
                    result: List[Tensor] = torch.ops.pyc_tensor.nusol_combined_cartesian_Nu(pmc_b, pmc_mu, met_xy, masses, sigma, null)
                    return {"NuVec" : result[0], "chi2" : result[1]}

                @torch.jit.script
                def NuNu(pmc_b1: Tensor, pmc_b2: Tensor, pmc_mu1: Tensor, pmc_mu2, met_xy: Tensor, masses: Tensor, null: float = 1e-10) -> Dict[str, Tensor]:
                        out: List[Tensor] = torch.ops.pyc_tensor.nusol_combined_cartesian_NuNu(pmc_b1, pmc_b2, pmc_mu1, pmc_mu2, met_xy, masses, null)
                        return {
                                "NuVec_1": out[0] , "NuVec_2": out[1], "Diagonal": out[2], "n_": out[3],
                                "H_perp_1" : out[4], "H_perp_2" : out[5], "NoSols" : out[6]
                        }

    @torch.jit.script
    class nusol:

        @torch.jit.script
        def BaseMatrix(pmc_b: Tensor, pmc_mu: Tensor, masses: Tensor) -> Tensor:
            return torch.ops.pyc_tensor.nusol_BaseMatrix(pmc_b, pmc_mu, masses)

        @torch.jit.script
        def Intersection(A: Tensor, B: Tensor, null: float = 1e-10) -> Tuple[Tensor, Tensor]:
            return torch.ops.pyc_tensor.nusol_Intersection(A, B, null)

        @torch.jit.script
        def Nu(pmc_b: Tensor, pmc_mu: Tensor, met_xy: Tensor, masses: Tensor, sigma: Tensor, null: float = 1e-10) -> List[Tensor]:
            return torch.ops.pyc_tensor.nusol_Nu(pmc_b, pmc_mu, met_xy, masses, sigma, null)

        @torch.jit.script
        def NuNu(pmc_b: Tensor, pmc_mu: Tensor, met_xy: Tensor, masses: Tensor, sigma: Tensor, null: float = 1e-10) -> List[Tensor]:
            return torch.ops.pyc_tensor.nusol_Nu(pmc_b, pmc_mu, met_xy, masses, sigma, null)

        @torch.jit.script
        def NuNu(pmc_b1: Tensor, pmc_b2: Tensor, pmc_l1: Tensor, pmc_l2: Tensor, met_xy: Tensor, masses: Tensor, null: float = 1e-10) -> List[Tensor]:
            return torch.ops.pyc_tensor.nusol_NuNu(pmc_b1, pmc_b2, pmc_l1, pmc_l2, met_xy, masses, null)

    @torch.jit.script
    class graph:

        @torch.jit.script
        def edge_aggregation(edge_index: Tensor, prediction: Tensor, node_feature: Tensor) -> Dict[int, Dict[str, Tensor]]:
            res: List[List[Tensor]] = torch.ops.pyc_tensor.graph_base_combined_edge_aggregation(edge_index, prediction, node_feature)
            idx: int = len(res)
            output: Dict[int, Dict[str, Tensor]] = {}
            for ix in range(idx): output[ix] = {"clusters": res[ix][0], "unique_sum": res[ix][1], "reverse_clusters": res[ix][2], "node_sum": res[ix][3]}
            return output

        @torch.jit.script
        def node_aggregation(edge_index: Tensor, prediction: Tensor, node_feature: Tensor) -> Dict[int, Dict[str, Tensor]]:
            res: List[List[Tensor]] = torch.ops.pyc_tensor.graph_base_combined_node_aggregation(edge_index, prediction, node_feature)
            idx: int = len(res)
            output: Dict[int, Dict[str, Tensor]] = {}
            for ix in range(idx): output[ix] = {"clusters": res[ix][0], "unique_sum": res[ix][1], "reverse_clusters": res[ix][2], "node_sum": res[ix][3]}
            return output

    @torch.jit.script
    class operators:

        @torch.jit.script
        def Dot(v1: Tensor, v2: Tensor) -> Tensor:
            return torch.ops.pyc_tensor.operators_Dot(v1, v2)

        @torch.jit.script
        def Mul(v1: Tensor, v2: Tensor) -> Tensor:
            return torch.ops.pyc_tensor.operators_Mul(v1, v2)

        @torch.jit.script
        def CosTheta(v1: Tensor, v2: Tensor) -> Tensor:
            return torch.ops.pyc_tensor.operators_CosTheta(v1, v2)

        @torch.jit.script
        def SinTheta(v1: Tensor, v2: Tensor) -> Tensor:
            return torch.ops.pyc_tensor.operators_SinTheta(v1, v2)

        @torch.jit.script
        def Rx(angle: Tensor) -> Tensor:
            return torch.ops.pyc_tensor.operators_Rx(angle)

        @torch.jit.script
        def Ry(angle: Tensor) -> Tensor:
            return torch.ops.pyc_tensor.operators_Ry(angle)

        @torch.jit.script
        def Rz(angle: Tensor) -> Tensor:
            return torch.ops.pyc_tensor.operators_Rz(angle)

        @torch.jit.script
        def CoFactors(matrix: Tensor) -> Tensor:
            return torch.ops.pyc_tensor.operators_CoFactors(matrix)

        @torch.jit.script
        def Determinant(matrix: Tensor) -> Tensor:
            return torch.ops.pyc_tensor.operators_Determinant(matrix)

        @torch.jit.script
        def Inverse(matrix: Tensor) -> Tensor:
            return torch.ops.pyc_tensor.operators_Inverse(matrix)

