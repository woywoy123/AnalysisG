# cython: linetrace=True

from libc.math cimport sin, cos, sinh, atan2, asinh

__all__ = ["Px", "Py", "Pz", "PT", "Phi", "Eta", "PxPyPzEMass", "deltaR"]

cpdef float Px(float pt, float phi):
    return pt*cos(phi)

cpdef float Py(float pt, float phi):
    return pt*sin(phi)

cpdef float Pz(float pt, float eta):
    return pt*sinh(eta)

cpdef float PT(float px, float py):
    return (px**2 + py**2)**0.5

cpdef float Phi(float px, float py):
    return atan2(py, px)

cpdef float Eta(float px, float py, float pz):
    cdef float pt = PT(px, py)
    return asinh(pz/pt)

cpdef float PxPyPzEMass(float px, float py, float pz, float e):
    cdef float s = e**2 - px**2 - py**2 - pz**2 
    if s < 0:
        return 0
    return s**0.5

cpdef float deltaR(float eta1, float eta2, float phi1, float phi2):
    return ((eta1 - eta2)**2 + (phi1 - phi2)**2)**0.5
