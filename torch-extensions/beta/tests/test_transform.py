import pyext
from common import *

def test_transform_px():
    d1 = create_tensor_cpu_1d()
    d1_cu = d1.to(device = "cuda")
    p1 = create_vector_polar(1, 2, 3, 4)
    assert rounder(pyext.Transform.Px(d1[:, 0], d1[:, 2]), p1.px)
    assert rounder(pyext.Transform.Px(d1), p1.px)
    assert rounder(pyext.Transform.Px(d1_cu[:, 0], d1_cu[:, 2]), p1.px)
    assert rounder(pyext.Transform.Px(d1_cu), p1.px)
    assert rounder(pyext.Transform.Px(1, 3), p1.px)
    assert rounder(pyext.Transform.Px([[1, 2, 3, 4]]), p1.px)

def test_transform_py():
    d1 = create_tensor_cpu_1d()
    d1_cu = d1.to(device = "cuda")
    p1 = create_vector_polar(1, 2, 3, 4)
    assert rounder(pyext.Transform.Py(d1[:, 0], d1[:, 2]), p1.py)
    assert rounder(pyext.Transform.Py(d1), p1.py)
    assert rounder(pyext.Transform.Py(d1_cu[:, 0], d1_cu[:, 2]), p1.py)
    assert rounder(pyext.Transform.Py(d1_cu), p1.py)
    assert rounder(pyext.Transform.Py(1, 3), p1.py)
    assert rounder(pyext.Transform.Py([[1, 2, 3, 4]]), p1.py)

def test_transform_pz():
    d1 = create_tensor_cpu_1d()
    d1_cu = d1.to(device = "cuda")
    p1 = create_vector_polar(1, 2, 3, 4)
    assert rounder(pyext.Transform.Pz(d1[:, 0], d1[:, 1]), p1.pz)
    assert rounder(pyext.Transform.Pz(d1), p1.pz)
    assert rounder(pyext.Transform.Pz(d1_cu[:, 0], d1_cu[:, 1]), p1.pz)
    assert rounder(pyext.Transform.Pz(d1_cu), p1.pz)
    assert rounder(pyext.Transform.Pz(1, 2), p1.pz)
    assert rounder(pyext.Transform.Pz([[1, 2, 3, 4]]), p1.pz)

def test_transform_pxpypz():
    d1 = create_tensor_cpu_1d()
    d1_cu = d1.to(device = "cuda")
    p1 = create_vector_polar(1, 2, 3, 4)
    p1 = [p1.px, p1.py, p1.pz]
    assert rounder_l(pyext.Transform.PxPyPz(d1[:, 0], d1[:, 1], d1[:, 2]), p1)
    assert rounder_l(pyext.Transform.PxPyPz(d1), p1)
    assert rounder_l(pyext.Transform.PxPyPz(d1_cu[:, 0], d1_cu[:, 1], d1_cu[:, 2]), p1)
    assert rounder_l(pyext.Transform.PxPyPz(d1_cu), p1)
    assert rounder_l(pyext.Transform.PxPyPz(1, 2, 3), p1)
    assert rounder_l(pyext.Transform.PxPyPz([[1, 2, 3, 4]]), p1)

def test_transform_pxpypze():
    d1 = create_tensor_cpu_1d()
    d1_cu = d1.to(device = "cuda")
    p1 = create_vector_polar(1, 2, 3, 4)
    p1 = [p1.px, p1.py, p1.pz, p1.e]
    assert rounder_l(pyext.Transform.PxPyPzE(d1[:, 0], d1[:, 1], d1[:, 2], d1[:, 3]), p1)
    assert rounder_l(pyext.Transform.PxPyPzE(d1), p1)
    assert rounder_l(pyext.Transform.PxPyPzE(d1_cu[:, 0], d1_cu[:, 1], d1_cu[:, 2], d1_cu[:, 3]), p1)
    assert rounder_l(pyext.Transform.PxPyPzE(d1_cu), p1)
    assert rounder_l(pyext.Transform.PxPyPzE(1, 2, 3, 4), p1)
    assert rounder_l(pyext.Transform.PxPyPzE([[1, 2, 3, 4]]), p1)

def test_transform_pt():
    d1 = create_tensor_cpu_1d()
    d1_cu = d1.to(device = "cuda")
    p1 = create_vector_cartesian(1, 2, 3, 4).pt
    assert rounder(pyext.Transform.Pt(d1[:, 0], d1[:, 1]), p1)
    assert rounder(pyext.Transform.Pt(d1), p1)
    assert rounder(pyext.Transform.Pt(d1_cu[:, 0], d1_cu[:, 1]), p1)
    assert rounder(pyext.Transform.Pt(d1_cu), p1)
    assert rounder(pyext.Transform.Pt(1, 2), p1)
    assert rounder(pyext.Transform.Pt([[1, 2, 3, 4]]), p1)

def test_transform_eta():
    d1 = create_tensor_cpu_1d()
    d1_cu = d1.to(device = "cuda")
    p1 = create_vector_cartesian(1, 2, 3, 4).eta
    assert rounder(pyext.Transform.Eta(d1[:, 0], d1[:, 1], d1[:, 2]), p1)
    assert rounder(pyext.Transform.Eta(d1), p1)
    assert rounder(pyext.Transform.Eta(d1_cu[:, 0], d1_cu[:, 1], d1_cu[:, 2]), p1)
    assert rounder(pyext.Transform.Eta(d1_cu), p1)
    assert rounder(pyext.Transform.Eta(1, 2, 3), p1)
    assert rounder(pyext.Transform.Eta([[1, 2, 3, 4]]), p1)

def test_transform_phi():
    d1 = create_tensor_cpu_1d()
    d1_cu = d1.to(device = "cuda")
    p1 = create_vector_cartesian(1, 2, 3, 4).phi
    assert rounder(pyext.Transform.Phi(d1[:, 0], d1[:, 1]), p1)
    assert rounder(pyext.Transform.Phi(d1), p1)
    assert rounder(pyext.Transform.Phi(d1_cu[:, 0], d1_cu[:, 1]), p1)
    assert rounder(pyext.Transform.Phi(d1_cu), p1)
    assert rounder(pyext.Transform.Phi(1, 2), p1)
    assert rounder(pyext.Transform.Phi([[1, 2, 3, 4]]), p1)

def test_transform_ptetaphi():
    d1 = create_tensor_cpu_1d()
    d1_cu = d1.to(device = "cuda")
    p1 = create_vector_cartesian(1, 2, 3, 4)
    p1 = [p1.pt, p1.eta, p1.phi]
    assert rounder_l(pyext.Transform.PtEtaPhi(d1[:, 0], d1[:, 1], d1[:, 2]), p1)
    assert rounder_l(pyext.Transform.PtEtaPhi(d1), p1)
    assert rounder_l(pyext.Transform.PtEtaPhi(d1_cu[:, 0], d1_cu[:, 1], d1_cu[:, 2]), p1)
    assert rounder_l(pyext.Transform.PtEtaPhi(d1_cu), p1)
    assert rounder_l(pyext.Transform.PtEtaPhi(1, 2, 3), p1)
    assert rounder_l(pyext.Transform.PtEtaPhi([[1, 2, 3, 4]]), p1)

def test_transform_ptetaphie():
    d1 = create_tensor_cpu_1d()
    d1_cu = d1.to(device = "cuda")
    p1 = create_vector_cartesian(1, 2, 3, 4)
    p1 = [p1.pt, p1.eta, p1.phi, p1.e]
    assert rounder_l(pyext.Transform.PtEtaPhiE(d1[:, 0], d1[:, 1], d1[:, 2], d1[:, 3]), p1)
    assert rounder_l(pyext.Transform.PtEtaPhiE(d1), p1)
    assert rounder_l(pyext.Transform.PtEtaPhiE(d1_cu[:, 0], d1_cu[:, 1], d1_cu[:, 2], d1_cu[:, 3]), p1)
    assert rounder_l(pyext.Transform.PtEtaPhiE(d1_cu), p1)
    assert rounder_l(pyext.Transform.PtEtaPhiE(1, 2, 3, 4), p1)
    assert rounder_l(pyext.Transform.PtEtaPhiE([[1, 2, 3, 4]]), p1)

if __name__ == "__main__":
    test_transform_px()
    test_transform_py()
    test_transform_pz()
    test_transform_pxpypz()
    test_transform_pxpypze()

    test_transform_pt()
    test_transform_eta()
    test_transform_phi()
    test_transform_ptetaphi()
    test_transform_ptetaphie()
