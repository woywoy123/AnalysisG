import torch

t = torch.tensor([1, 1, 1, 1], device = "cuda", dtype = torch.float64)
def loads(inpt = "../build/transform/libTransformCuda.so"): torch.ops.load_library(inpt)
def get(libs, out): return getattr(getattr(torch.ops, libs), out)
def px(): px = get("TransformCuda", "Px")(t, t)
def py(): py = get("TransformCuda", "Py")(t, t)
def pz(): pz = get("TransformCuda", "Pz")(t, t)
def pxpypz(): pxpypz = get("TransformCuda", "PxPyPz")(t, t, t)
def pxpypze():
    t = torch.tensor([1, 1, 1, 1], device = "cuda", dtype = torch.float64).view(-1, 1)
    pxpypzE = get("TransformCuda", "PxPyPzE")(torch.cat([t, t, t, t], -1))

if __name__ == "__main__":
    loads()
    px()
    py()
    pz()
    pxpypz()
    pxpypze()
