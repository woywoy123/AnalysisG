import torch

t = 1
def loads(inpt = "../build/libTransformFloats.so"): torch.ops.load_library(inpt)
def get(libs, out): return getattr(getattr(torch.ops, libs), out)
def pt(): px = get("TransformFloats", "Pt")(t, t)
def eta(): py = get("TransformFloats", "Eta")(t, t, t)
def phi(): pz = get("TransformFloats", "Phi")(t, t)
def ptetaphi(): pxpypz = get("TransformFloats", "PtEtaPhi")(t, t, t)

if __name__ == "__main__":
    loads()
    pt()
    eta()
    phi()
    ptetaphi()

