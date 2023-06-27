import pyext
import torch

def create_tensor_cpu_1d():
    return torch.tensor([1., 1., 1., 1.]).view(-1, 1)

def create_tensor_cpu_nd(dim):
    return torch.cat([create_tensor_cpu_1d for _ in range(dim)], -1)

def test_transform_px():
    d1 = create_tensor_cpu_1d()
    d1 = d1
    print(d1)
    print(pyext.Transform.Px(d1, d1))

if __name__ == "__main__":
    test_transform_px()
