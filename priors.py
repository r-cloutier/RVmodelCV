import numpy as np

def P_prior(P, Pmin=1, Pmax=1e4):
    Pmin, Pmax = float(Pmin), float(Pmax)
    return 1. / (P*np.log(Pmax/Pmin))

def K_prior(K, Kmax=999, K0=1):
    Kmax, K0 = float(Kmax), float(K0)
    return 1. / (K0*(1+K/K0) * np.log(1+Kmax/K0))

def e_prior(e):
    return 1.

def omega_prior(omega):
    return 1. / (2*np.pi)

def M_prior(M):  # mean anomaly
    '''M = n*(t-T0) = 2*pi*(t-T0)/P'''
    return 1. / (2*np.pi)

def jitter_prior(sigma, sigmamax=99, sigma0=1):
    return 1. / (sigma0*(1+sigma/sigma0) * np.log(1+sigmamax/sigma0))

def C_prior(C, Cmax=1e3):
    return 1./(2*Cmax) if abs(C) <= Cmax else 0.

def planet_prior(Nplanets, alpha=1./3):
    if Nplanets == 0:
        return 1.-np.sum([alpha**i for i in range(1,4)])
    elif 1 <= Nplanets <= 3:
        return alpha**Nplanets
    else:
        return 0.
