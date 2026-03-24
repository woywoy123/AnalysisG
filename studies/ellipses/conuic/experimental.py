from constants import *
import numpy as np

def Lines(data, m_nu, tau, phi, ws, dts, eps):
    w, O = omega(data, ws), Omega(data, ws)
    dt = delta(data, dts)
    
    kappa = np.atan(w)
    c = (dt * w * data.e_mu**2 - data.m_mu**2) / data.p_mu
    d1 = eps * O * (np.cos(kappa) - dt * np.sin(kappa)) / data.b_mu
    d2 = (np.sin(kappa) + dt * np.cos(kappa))
    return c + m_nu * (d1 * np.cosh(tau) - d2 * np.sinh(tau) * np.cos(phi))

def m_nufx(data, tau, phi, sw, sd):
    dt = delta(data, sd)
    w, O = omega(data, sw), Omega(data, sw)
    e_mu, m_mu, b_mu = data.e_mu, data.m_mu, data.b_mu

    kappa = np.atan(w)
    sk, ck, tk = np.sin(kappa), np.cos(kappa), np.tan(kappa)
    n1 = dt * tk * e_mu ** 2 - m_mu ** 2
    d11 = O * np.cosh(tau) * (dt * tk - 1) 
    d12 = b_mu * np.cos(phi) * b_mu * np.sinh(tau) * (dt + tk)
    return n1 / (e_mu * ck * ( d11 + d12 ) )

def m_nutau(data, phi, sw, sd):
    O, w = Omega(data, sw), omega(data, sw)
    dt = delta(data, sd)
    b_mu = data.b_mu
    d1 = O * dt * w  + O + b_mu * dt * np.cos(phi) + b_mu * np.cos(phi) * w
    d2 = O * dt * w  - O + b_mu * dt * np.cos(phi) + b_mu * np.cos(phi) * w
    if d1 / d2 < 0: return np.log(-d2 / d1) * 0.5
    return np.log(d1 / d2) * 0.5

def dLdtau(data, m_nu, tau, phi, ws, dts, eps):
    w, O = omega(data, ws), Omega(data, ws)
    dt = delta(data, dts)
    
    kappa = np.atan(w)
    d1 = eps * O * (np.cos(kappa) - dt * np.sin(kappa)) / data.b_mu
    d2 = (np.sin(kappa) + dt * np.cos(kappa))
    return m_nu * (d1 * np.sinh(tau) - d2 * np.cosh(tau) * np.cos(phi))


def m_nuphi(data, sw, sd):
    O, w = Omega(data, sw), omega(data, sw)
    dt = delta(data, sd)
    b_mu = data.b_mu
    s = O * (dt * w - 1) / (b_mu * (dt + w))
    if abs(s) > 1: return np.acos(1 / s)
    return np.acos(s)

def m2_nu(data, s1, s2):
    #note: branch independent of s1 -> only delta matters
    dt, w, O = delta(data, s2), omega(data, s1), Omega(data, s1)
    a = (data.p_mu * dt * O) ** 2
    b = (dt + w)**2 - O**2 * (dt**2 + 1)
    return (data.m_mu ** 2 - a / b)**0.5

# -------------- phi and tau maxima ---------------- #
def m_Nux(data, ws, ds):
    dt = delta(data, ds)
    kappa = np.atan(omega(data, ds))

    n1 = data.m_mu ** 2 - dt * np.tan(kappa) * data.e_mu ** 2
    d1 = Omega(data, ds) ** 2 * (np.cos(kappa) - dt * np.sin(kappa)) ** 2  
    d2 =       data.b_mu ** 2 * (np.sin(kappa) + dt * np.cos(kappa)) ** 2
    return n1 / (data.e_mu * (d1 - d2) ** 0.5)


def tau_H(data, ws, ds):
    O, w, dt = Omega(data, ws), omega(data, ws), delta(data, ds)
    kappa = np.atan(w)
    t = (data.b_mu / O) * (np.sin(kappa) + dt * np.cos(kappa)) / (np.cos(kappa) - dt * np.sin(kappa))
    return np.atanh(t)


