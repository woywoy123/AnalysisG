import torch

t = torch.tensor([1, 1, 1, 1], device = "cuda", dtype = torch.float64)
def loads(inpt = "../build/libTransformTensors.so"): torch.ops.load_library(inpt)
def get(libs, out): return getattr(getattr(torch.ops, libs), out)
def px(): px = get("TransformTensors", "Px")(t, t)
def py(): py = get("TransformTensors", "Py")(t, t)
def pz(): pz = get("TransformTensors", "Pz")(t, t)
def pxpypz(): pxpypz = get("TransformTensors", "PxPyPz")(t, t, t)
def pxpypze():
    t = torch.tensor([1, 1, 1, 1], device = "cuda", dtype = torch.float64).view(-1, 4)
    t = torch.cat([t, t*2, t*3], 0)
    pxpypzE = get("TransformTensors", "PxPyPzE")(t)

if __name__ == "__main__":
    loads()
    px()
    py()
    pz()
    pxpypz()
    pxpypze()
