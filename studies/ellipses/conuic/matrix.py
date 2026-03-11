from constants import *
from classes import *
from atomics import *
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize, linewidth = 100000, precision = 100)


def H_tilde(sx, sy, sz, data, sign):
    x = np.array([
        [sz / Omega(data, sign),                     0, x1(sx, sy, data, sign) - data.p_mu], 
        [omega(data, sign) * sz / Omega(data, sign), 0, y1(sx, sy, data, sign)            ],
        [0                                         , sz, 0                                ]
    ], np.complex128)
    return x

def M_matrix(data, s1):
    O, w = Omega(data, s1), omega(data, s1)
    return np.array([
        [1 - 1 / O**2,     - w / O**2],
        [- w / O**2  , 1 - (w / O)**2]
    ])

