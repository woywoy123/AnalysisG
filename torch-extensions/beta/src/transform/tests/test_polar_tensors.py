import torch

t = torch.tensor([1, 1, 1, 1], device = "cuda", dtype = torch.float64)
def loads(inpt = "../build/libTransformTensors.so"): torch.ops.load_library(inpt)
def get(libs, out): return getattr(getattr(torch.ops, libs), out)
def pt(): px = get("TransformTensors", "Pt")(t, t)
def eta(): py = get("TransformTensors", "Eta")(t, t, t)
def phi(): pz = get("TransformTensors", "Phi")(t, t)
def ptetaphi(): pxpypz = get("TransformTensors", "PtEtaPhi")(t, t, t)
def ptetaphie():
    t = torch.tensor([1, 1, 1, 1], device = "cuda", dtype = torch.float64).view(-1, 1)
    pxpypzE = get("TransformTensors", "PtEtaPhiE")(torch.cat([t, t, t, t], -1))


if __name__ == "__main__":
    loads()
    pt()
    eta()
    phi()
    ptetaphi()
    ptetaphie()
