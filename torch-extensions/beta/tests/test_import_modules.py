import torch

root = "../build/transform/lib"
t = torch.tensor([1, 1, 1, 1], device = "cuda", dtype = torch.float64)

def loads(inpt): torch.ops.load_library(inpt)
def get(libs, out):
    loads(root + libs + ".so")
    return getattr(getattr(torch.ops, libs), out)

def test_cuda_px():           get("TransformCuda", "Px")(t, t)
def test_cuda_py():           get("TransformCuda", "Py")(t, t)
def test_cuda_pz():           get("TransformCuda", "Pz")(t, t)
def test_cuda_pxpypz():       get("TransformCuda", "PxPyPz")(t, t, t)
def test_cuda_pxpypze():
    x = torch.tensor([1, 1, 1, 1], device = "cuda", dtype = torch.float64).view(-1, 1)
    get("TransformCuda", "PxPyPzE")(torch.cat([x, x, x, x], -1))

def test_cuda_pt():           get("TransformCuda", "Pt")(t, t)
def test_cuda_eta():          get("TransformCuda", "Eta")(t, t, t)
def test_cuda_phi():          get("TransformCuda", "Phi")(t, t)
def test_cuda_ptetaphi():     get("TransformCuda", "PtEtaPhi")(t, t, t)
def test_cuda_ptetaphie():
    x = torch.tensor([1, 1, 1, 1], device = "cuda", dtype = torch.float64).view(-1, 1)
    get("TransformCuda", "PtEtaPhiE")(torch.cat([x, x, x, x], -1))

def test_tensor_px():         get("TransformTensors", "Px")(t, t)
def test_tensor_py():         get("TransformTensors", "Py")(t, t)
def test_tensor_pz():         get("TransformTensors", "Pz")(t, t)
def test_tensor_pxpypz():     get("TransformTensors", "PxPyPz")(t, t, t)
def test_tensor_pxpypze():
    x = torch.tensor([1, 1, 1, 1], device = "cuda", dtype = torch.float64).view(-1, 4)
    x = torch.cat([x, x*2, x*3], 0)
    get("TransformTensors", "PxPyPzE")(x)

def test_tensor_pt():         get("TransformTensors", "Pt")(t, t)
def test_tensor_eta():        get("TransformTensors", "Eta")(t, t, t)
def test_tensor_phi():        get("TransformTensors", "Phi")(t, t)
def test_tensor_ptetaphi():   get("TransformTensors", "PtEtaPhi")(t, t, t)
def test_tensor_ptetaphie():
    x = torch.tensor([1, 1, 1, 1], device = "cuda", dtype = torch.float64).view(-1, 1)
    get("TransformTensors", "PtEtaPhiE")(torch.cat([x, x, x, x], -1))

def test_float_px():       get("TransformFloats", "Px")(1, 1)
def test_float_py():       get("TransformFloats", "Py")(1, 1)
def test_float_pz():       get("TransformFloats", "Pz")(1, 1)
def test_float_pxpypz():   get("TransformFloats", "PxPyPz")(1, 1, 1)

def test_float_pt():        get("TransformFloats", "Pt")(1, 1)
def test_float_eta():       get("TransformFloats", "Eta")(1, 1, 1)
def test_float_phi():       get("TransformFloats", "Phi")(1, 1)
def test_float_ptetaphi():  get("TransformFloats", "PtEtaPhi")(1, 1, 1)

if __name__ == "__main__":
    test_float_px()
    test_float_py()
    test_float_pz()
    test_float_pxpypz()

    test_float_pt()
    test_float_eta()
    test_float_phi()
    test_float_ptetaphi()

    test_cuda_px()
    test_cuda_py()
    test_cuda_pz()
    test_cuda_pxpypz()
    test_cuda_pxpypze()

    test_cuda_pt()
    test_cuda_eta()
    test_cuda_phi()
    test_cuda_ptetaphi()
    test_cuda_ptetaphie()

    test_tensor_px()
    test_tensor_py()
    test_tensor_pz()
    test_tensor_pxpypz()
    test_tensor_pxpypze()

    test_tensor_pt()
    test_tensor_eta()
    test_tensor_phi()
    test_tensor_ptetaphi()
    test_tensor_ptetaphie()




