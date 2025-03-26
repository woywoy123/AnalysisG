import torch

class pyc:

    def __init__(self, devices = ["cupyc"]): #, "tpyc"]):
        self._pth = "../build/pyc/interface"
        self._fx = ["graph", "nusol", "operators", "physics", "transform"]
        self._ops = {
                "nusol" : [
                    "base_basematrix", "nu", "nunu", "combinatorial"
                ],
                "operators" : [
                   "dot"      , "eigenvalue" , "costheta" , "sintheta"   ,
                   "rx"       , "ry"         , "rz"       , "rt"         ,
                   "cofactors", "determinant", "inverse"  , "cross"
                ],
                "physics" : [
                    "cartesian_separate_p2"    , "polar_combined_deltaR"    ,
                    "cartesian_combined_p2"    , "polar_separate_p2"        ,
                    "cartesian_separate_p"     , "cartesian_combined_p"     ,
                    "polar_separate_p"         , "polar_combined_p"         ,
                    "cartesian_separate_beta2" , "cartesian_combined_beta2" ,
                    "polar_separate_beta2"     , "polar_combined_beta2"     ,
                    "cartesian_separate_beta"  , "cartesian_combined_beta"  ,
                    "polar_separate_beta"      , "polar_combined_beta"      ,
                    "cartesian_separate_m2"    , "cartesian_combined_m2"    ,
                    "polar_separate_m2"        , "polar_combined_m2"        ,
                    "cartesian_separate_m"     , "cartesian_combined_m"     ,
                    "polar_separate_m"         , "polar_combined_m"         ,
                    "cartesian_separate_mt2"   , "cartesian_combined_mt2"   ,
                    "polar_combined_p2"        , "polar_separate_mt2"       ,
                    "polar_combined_mt2"       , "cartesian_separate_mt"    ,
                    "cartesian_combined_mt"    , "polar_separate_mt"        ,
                    "polar_combined_mt"        , "cartesian_separate_theta" ,
                    "cartesian_combined_theta" , "polar_separate_theta"     ,
                    "polar_combined_theta"     , "cartesian_separate_deltaR",
                    "cartesian_combined_deltaR", "polar_separate_deltaR"
                ],
                "transform" : [
                    "separate_px"       ,
                    "separate_py"       , "separate_pz"            ,
                    "separate_pxpypz"   , "separate_pxpypze"       ,
                    "combined_px"       , "combined_py"            ,
                    "combined_pz"       , "combined_pxpypz"        ,
                    "combined_pxpypze"  , "separate_pt"            ,
                    "separate_phi"      , "separate_eta"           ,
                    "separate_ptetaphi" , "separate_ptetaphie"     ,
                    "combined_pt"       , "combined_phi"           ,
                    "combined_eta"      , "combined_ptetaphi"      ,
                    "combined_ptetaphie"
                ],
                "graph" : ["edge_aggregation", "node_aggregation", "unique_aggregation", "page_rank"]
        }


        for i in devices: torch.ops.load_library(self._pth + "/lib" + i + ".so")
        lx =  [(i, i + "_" + j) for i in self._fx for j in self._ops[i]]
        for d in devices:
            for l in lx:
                op, fx = l
                try: setattr(self, d + "_" + fx, getattr(getattr(torch.ops, d), fx))
                except AttributeError: pass
