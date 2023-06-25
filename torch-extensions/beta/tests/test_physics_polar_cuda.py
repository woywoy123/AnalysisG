import torch

t = torch.tensor([2, 1, 1, 1], device = "cuda", dtype = torch.float64)
def loads(inpt = "../build/physics/libPhysicsPolar.so"): torch.ops.load_library(inpt)
def get(libs, out): return getattr(getattr(torch.ops, libs), out)
def test_p2():
    print(get("PhysicsPolar", "P2")(t, t, t))

if __name__ == "__main__":
    loads()
    test_p2()
