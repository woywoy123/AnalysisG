from common import *
from nusol import *
import random
import torch
import math
import time

torch.ops.load_library("../build/pyc/interface/libcupyc.so")
device = "cuda"
torch.set_printoptions(threshold=1000000)

def checkthis(data, cu, cufx, fx, cart = True):
    def lst(tmp): return [fx(p) for p in tmp]
    dx = "physics" + "_" + ("cartesian" if cart else "polar") + "_"
    sep = dx + "separate" + "_" + fx.__name__[1:]
    cmp = dx + "combined" + "_" + fx.__name__[1:]
    test = lst(data)
    print(sep)
    assert rounder_l(getattr(torch.ops.cupyc, sep)(*cufx(cu)), test)
    print(cmp)
    assert rounder_l(getattr(torch.ops.cupyc, cmp)(cu), test)


def test_transform():
    test_case = [random.random() for i in range(4)]
    p1 = create_vector_cartesian(*test_case)
    d1_cu = create_tensor_cpu_1d(test_case).to(device = device)

    assert rounder(torch.ops.cupyc.transform_separate_pt(d1_cu[:, 0], d1_cu[:, 1]), p1.pt)
    assert rounder(torch.ops.cupyc.transform_combined_pt(d1_cu), p1.pt)
    assert rounder(torch.ops.cupyc.transform_separate_phi(d1_cu[:, 0], d1_cu[:, 1]), p1.phi)
    assert rounder(torch.ops.cupyc.transform_combined_phi(d1_cu), p1.phi)
    assert rounder(torch.ops.cupyc.transform_separate_eta(d1_cu[:, 0], d1_cu[:, 1], d1_cu[:, 2]), p1.eta)
    assert rounder(torch.ops.cupyc.transform_combined_eta(d1_cu), p1.eta)
    assert rounder_l(torch.ops.cupyc.transform_separate_ptetaphi(d1_cu[:, 0], d1_cu[:, 1], d1_cu[:, 2]), [p1.pt, p1.eta, p1.phi])
    assert rounder_l(torch.ops.cupyc.transform_combined_ptetaphi(d1_cu), [p1.pt, p1.eta, p1.phi])
    assert rounder_l(torch.ops.cupyc.transform_separate_ptetaphie(d1_cu[:, 0], d1_cu[:, 1], d1_cu[:, 2], d1_cu[:, 3]), [p1.pt, p1.eta, p1.phi])
    assert rounder_l(torch.ops.cupyc.transform_combined_ptetaphie(d1_cu), [p1.pt, p1.eta, p1.phi])

    p1 = create_vector_polar(*test_case)
    assert rounder(torch.ops.cupyc.transform_separate_px(d1_cu[:, 0], d1_cu[:, 2]), p1.px)
    assert rounder(torch.ops.cupyc.transform_combined_px(d1_cu), p1.px)
    assert rounder(torch.ops.cupyc.transform_separate_py(d1_cu[:, 0], d1_cu[:, 2]), p1.py)
    assert rounder(torch.ops.cupyc.transform_combined_py(d1_cu), p1.py)
    assert rounder(torch.ops.cupyc.transform_separate_pz(d1_cu[:, 0], d1_cu[:, 1]), p1.pz)
    assert rounder(torch.ops.cupyc.transform_combined_pz(d1_cu), p1.pz)
    assert rounder_l(torch.ops.cupyc.transform_separate_pxpypz(d1_cu[:, 0], d1_cu[:, 1], d1_cu[:, 2]), [p1.px, p1.py, p1.pz])
    assert rounder_l(torch.ops.cupyc.transform_combined_pxpypz(d1_cu), [p1.px, p1.py, p1.pz])
    assert rounder_l(torch.ops.cupyc.transform_separate_pxpypze(d1_cu[:, 0], d1_cu[:, 1], d1_cu[:, 2], d1_cu[:, 3]), [p1.px, p1.py, p1.pz])
    assert rounder_l(torch.ops.cupyc.transform_combined_pxpypze(d1_cu), [p1.px, p1.py, p1.pz])



if __name__ == "__main__":
    test_transform()










