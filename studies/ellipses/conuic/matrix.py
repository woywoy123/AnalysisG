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

def H_tildeP(m_nu, tau, phi, data, sign, eps):
    O, w = Omega(data, sign), omega(data, sign)
    kappa = np.atan(w)

    a00 = (1 / O) * np.sinh(tau) * np.sin(phi)
    a10 = (np.tan(kappa) / O) * np.sinh(tau) * np.sin(phi)
    a21 = np.sinh(tau) * np.sin(phi)

    a02 = -((eps * data.b_mu / O) * np.cosh(tau) * np.cos(kappa) + np.sin(kappa) * np.sinh(tau) * np.cos(phi))
    a12 =  -(data.b_mu / O) * np.cosh(tau) * np.sin(kappa) + np.cos(kappa) * np.sinh(tau) * np.cos(phi)
   
    x = np.array([
        [a00, 0  , a02], 
        [a10, 0  , a12],
        [0  , a21, 0  ]
    ])
    if np.isnan(np.linalg.det(x)): return None
    return m_nu * x

