# cython: linetrace=True

from libc.math cimport sin, cos, sinh, atan2, asinh, pow

__all__ = ["Px", "Py", "Pz", "PT", "Phi", "Eta", "PxPyPzEMass", "deltaR", "energy", "IsIn"]

cpdef float Px(float pt, float phi):
    return pt*cos(phi)

cpdef float Py(float pt, float phi):
    return pt*sin(phi)

cpdef float Pz(float pt, float eta):
    return pt*sinh(eta)

cpdef float PT(float px, float py):
    return pow( pow(px, 2) + pow(py, 2), 0.5 )

cpdef float Phi(float px, float py):
    return atan2(py, px)

cpdef float Eta(float px, float py, float pz):
    cdef float pt = PT(px, py)
    return asinh(pz/pt)

cpdef float PxPyPzEMass(float px, float py, float pz, float e):
    cdef float s = pow(e, 2) - pow(px, 2) - pow(py, 2) - pow(pz, 2)
    if s < 0:
        return 0
    return pow(s, 0.5)/1000

cpdef float deltaR(float eta1, float eta2, float phi1, float phi2):
    return pow(pow(eta1 - eta2, 2) + pow(phi1 - phi2, 2), 0.5)

cpdef float energy(float m, float px, float py, float pz):
    cdef float p2 = pow(px, 2) + pow(py, 2) + pow(pz, 2)
    return pow( p2 + pow(m, 2), 0.5 )

cpdef IsIn(list srch, dict objdic):
    cdef str l
    cdef int cnt = 0
    for l in srch:
        if l in objdic: 
            cnt += 1
    if len(srch) == cnt:
        return True
    return False
