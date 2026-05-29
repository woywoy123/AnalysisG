from numba import jit, prange, njit
import numpy as np

def FT(x, y, z, vl, vp, Q):
    x0 =  np.zeros_like(x)
    for i in prange(len(x)): 
        for j in prange(len(x[i])):
            for k in prange(len(x[i][j])):
                vX = np.array([x[i][j][k], y[i][j][k], z[i][j][k], 1.0], dtype = np.float64)
                v, v_  = (vp - vX), (vl - vX)
                x0[i][j][k] = (v.dot(Q).dot(v_.T)).reshape(-1)[0]
    return x0

@njit(parallel = True)
def FX(x, y, z, Q):
    G = np.zeros_like(x)
    for i in prange(len(x)): 
        for j in prange(len(x[i])):
            for k in prange(len(x[i][j])):
                S = np.array([x[i][j][k], y[i][j][k], z[i][j][k], 1])
                G[i][j][k] = S.dot(Q).dot(S.T)
    return G

#@njit(parallel = True)
def FXR(x, y, z, R, Q):
    G = np.zeros_like(x)
    f = R.T.dot(Q).dot(R)
    for i in prange(len(x)): 
        for j in prange(len(x[i])):
            for k in prange(len(x[i][j])):
                v = np.array([x[i][j][k], y[i][j][k], z[i][j][k], 1.0], dtype = np.float64).reshape((-1))
                G[i][j][k] = v.dot(f).dot(v.T).flatten()[0]
    return G

def ShiftPQl(x, y, z, Q, B):
    G = FX(x, y, z, Q)
    for i in prange(len(x)): 
        for j in prange(len(x[i])):
            for k in prange(len(x[i][j])):
                v = np.array([1, 1, 1, 1])
                G[i][j][k] = v.dot(B(x[i][j][k], y[i][j][k], G[i][j][k]).dot(v))
    return G

def ShiftPQl(x, y, z, Q, B):
    G = FX(x, y, z, Q)
    for i in prange(len(x)): 
        for j in prange(len(x[i])):
            for k in prange(len(x[i][j])):
                v = np.array([1, 1, 1, 1])
                G[i][j][k] = v.dot(B(x[i][j][k], y[i][j][k], G[i][j][k]).dot(v))
    return G










