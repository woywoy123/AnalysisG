import pyext.Physics as Physics
from common import *

def test_polar_p2():
    d1 = create_tensor_cpu_1d()
    d1_cu = d1.to(device = "cuda")
    p1 = create_vector_polar(1, 2, 3, 4).p2
    assert rounder(Physics.Polar.P2(d1_cu[:, 0], d1_cu[:, 1], d1_cu[:, 2]), p1)
    assert rounder(Physics.Polar.P2(d1_cu), p1)

def test_polar_p():
    d1 = create_tensor_cpu_1d()
    d1_cu = d1.to(device = "cuda")
    p1 = create_vector_polar(1, 2, 3, 4).p
    assert rounder(Physics.Polar.P(d1_cu[:, 0], d1_cu[:, 1], d1_cu[:, 2]), p1)
    assert rounder(Physics.Polar.P(d1_cu), p1)

def test_cartesian_p2():
    d1 = create_tensor_cpu_1d()
    d1_cu = d1.to(device = "cuda")
    p1 = create_vector_cartesian(1, 2, 3, 4).p2
    assert rounder(Physics.Cartesian.P2(d1_cu[:, 0], d1_cu[:, 1], d1_cu[:, 2]), p1)
    assert rounder(Physics.Cartesian.P2(d1_cu), p1)

def test_cartesian_p():
    d1 = create_tensor_cpu_1d()
    d1_cu = d1.to(device = "cuda")
    p1 = create_vector_cartesian(1, 2, 3, 4).p
    assert rounder(Physics.Cartesian.P(d1_cu[:, 0], d1_cu[:, 1], d1_cu[:, 2]), p1)
    assert rounder(Physics.Cartesian.P(d1_cu), p1)


def test_polar_beta2():
    d1 = create_tensor_cpu_1d()
    d1_cu = d1.to(device = "cuda")
    p1 = create_vector_polar(1, 2, 3, 4).beta**2
    assert rounder(Physics.Polar.Beta2(d1_cu[:, 0], d1_cu[:, 1], d1_cu[:, 2], d1_cu[:, 3]), p1)
    assert rounder(Physics.Polar.Beta2(d1_cu), p1)

def test_polar_beta():
    d1 = create_tensor_cpu_1d()
    d1_cu = d1.to(device = "cuda")
    p1 = create_vector_polar(1, 2, 3, 4).beta
    assert rounder(Physics.Polar.Beta(d1_cu[:, 0], d1_cu[:, 1], d1_cu[:, 2], d1_cu[:, 3]), p1)
    assert rounder(Physics.Polar.Beta(d1_cu), p1)

def test_cartesian_beta2():
    d1 = create_tensor_cpu_1d()
    d1_cu = d1.to(device = "cuda")
    p1 = create_vector_cartesian(1, 2, 3, 4).beta**2
    assert rounder(Physics.Cartesian.Beta2(d1_cu[:, 0], d1_cu[:, 1], d1_cu[:, 2], d1_cu[:, 3]), p1)
    assert rounder(Physics.Cartesian.Beta2(d1_cu), p1)

def test_cartesian_beta():
    d1 = create_tensor_cpu_1d()
    d1_cu = d1.to(device = "cuda")
    p1 = create_vector_cartesian(1, 2, 3, 4).beta
    assert rounder(Physics.Cartesian.Beta(d1_cu[:, 0], d1_cu[:, 1], d1_cu[:, 2], d1_cu[:, 3]), p1)
    assert rounder(Physics.Cartesian.Beta(d1_cu), p1)


if __name__ == "__main__":
    test_polar_p2()
    test_polar_p()
    test_cartesian_p2()
    test_cartesian_p()

    test_polar_beta2()
    test_polar_beta()
    test_cartesian_beta2()
    test_cartesian_beta()
