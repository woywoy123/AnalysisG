def post_install_extensions():
        def fileH(key, head, lst, _ver):
                return { key + v : [ head + j + v + ".h" for j in lst ] for v in _ver }
        def fileCpp(key, head, lst, _ver):
                return { key + v : [ head + j + v + ".cxx" for j in lst ] + [head.replace("CXX", "Shared") + v + ".cxx"] for v in _ver }




        import torch
        from torch.utils.cpp_extension import BuildExtension, CppExtension
        ver = ["Floats", "Tensors", "CUDA"]
        _TransformH, _TransformC = {}, {}
        _TransformH |= fileH("PyC.Transform.", "src/Transform/Headers/" , ["ToCartesian", "ToPolar"], ver)
        _TransformC |= fileCpp("PyC.Transform.", "src/Transform/CXX/", ["ToCartesian", "ToPolar"], ver)
        
        _OperatorsH, _OperatorsC = {}, {}
        _TransformH |= fileH("PyC.Transform.", "src/Transform/Headers/" , ["ToCartesian", "ToPolar"], ver)
        _TransformC |= fileCpp("PyC.Transform.", "src/Transform/CXX/", ["ToCartesian", "ToPolar"], ver)
 

        print(_TransformC)



post_install_extensions()
