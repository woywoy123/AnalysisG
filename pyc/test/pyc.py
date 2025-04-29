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
        self.devices = devices
        self._modules = {}
        self._load_modules()

        for i in devices: torch.ops.load_library(self._pth + "/lib" + i + ".so")
        lx =  [(i, i + "_" + j) for i in self._fx for j in self._ops[i]]
        for d in devices:
            for l in lx:
                op, fx = l
                try: setattr(self, d + "_" + fx, getattr(getattr(torch.ops, d), fx))
                except AttributeError: pass

    def _load_modules(self):
        """Load all required modules based on the specified devices"""
        import importlib.util
        import os
        import sys
        
        for device in self.devices:
            for fx in self._fx:
                module_path = os.path.join(self._pth, device, f"{fx}.so")
                if os.path.exists(module_path):
                    spec = importlib.util.spec_from_file_location(f"{device}_{fx}", module_path)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[spec.name] = module
                    spec.loader.exec_module(module)
                    
                    if device not in self._modules:
                        self._modules[device] = {}
                    
                    self._modules[device][fx] = module
                else:
                    print(f"Warning: Module {module_path} not found")
                    
    def get_device_modules(self, device=None):
        """Return modules for the specified device or for all devices"""
        if device:
            return self._modules.get(device, {})
        return self._modules
        
    def run(self, operation, device=None, **kwargs):
        """Execute an operation on the specified device"""
        # Use the first available device if none specified
        if device is None:
            device = self.devices[0]
            
        # Extract function category and name
        if "." in operation:
            category, func_name = operation.split(".", 1)
        else:
            # Try to find the category based on the operation name
            category = None
            for cat, ops in self._ops.items():
                if operation in ops:
                    category = cat
                    func_name = operation
                    break
            
            if category is None:
                raise ValueError(f"Operation {operation} not found in any category")
        
        if category not in self._modules[device]:
            raise ValueError(f"Category {category} not available for device {device}")
            
        # Get the module
        module = self._modules[device][category]
        
        # Get the function
        if not hasattr(module, func_name):
            raise ValueError(f"Function {func_name} not found in {category} for device {device}")
            
        func = getattr(module, func_name)
        
        # Execute the function with the provided arguments
        return func(**kwargs)
    
    def list_operations(self, category=None):
        """List available operations, optionally filtered by category"""
        if category:
            return self._ops.get(category, [])
        return self._ops
        
    def get_devices(self):
        """Return list of available devices"""
        return list(self._modules.keys())
