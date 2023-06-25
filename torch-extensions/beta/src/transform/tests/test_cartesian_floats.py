import torch

def loads(inpt = "../build/libTransformFloats.so"): torch.ops.load_library(inpt)
def get(libs, out): return getattr(getattr(torch.ops, libs), out)

def px(): px = get("TransformFloats", "Px")(1, 1)
def py(): py = get("TransformFloats", "Py")(1, 1)
def pz(): pz = get("TransformFloats", "Pz")(1, 1)
def pxpypz(): pxpypz = get("TransformFloats", "PxPyPz")(1, 1, 1)

if __name__ == "__main__":
    loads()
    px()
    py()
    pz()
    pxpypz()
