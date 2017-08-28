from imports import *


def _param_conditions():
    '''
    Return the limits of the parameter values to map input parameters in physical units 
    to the interval [0,1].
    '''
    sigmaJrange = 1, 99
    Crange = -1e3, 1e3
    Prange = 1.25, 1e4
    Mrange = 0, 2*np.pi
    Krange = 0, 999
    erange = 0, 1
    omegarange = 0, 2*np.pi
    return sigmaJrange, Crange, Prange, Mrange, Krange, erange, omegarange


def precondition_data(x, A, B):
    A = float(A)
    return -(x - A) / (A - B)

def recondition_data(x, A, B):
    A = float(A)
    return -1 * x * (A - B) + A


def P_prior(P):
    _,_,Plims,_,_,_,_ = _param_conditions()
    Pmin, Pmax = float(Plims[0]), float(Plims[1])
    return 1. / (P*np.log(Pmax/Pmin))

def K_prior(K, K0=1):
    _,_,_,_,Klims,_,_ = _param_conditions()
    Kmax, K0 = float(Klims[1]), float(K0)
    return 1. / (K0*(1+K/K0) * np.log(1+Kmax/K0))

def e_prior(e, sigma_e=.2):
    e = float(e)
    PDF_Rayleigh = e/sigma_e**2 * np.exp(-.5*(e/sigma_e)**2)
    CDF_Rayleigh = 1. - np.exp(-.5*(e/sigma_e)**2)
    return PDF_Rayleigh / CDF_Rayleigh if 0 <= e <= 1 else 0
        
def omega_prior():
    return 1. / (2*np.pi)

def M_prior():  # mean anomaly
    '''M = n*(t-T0) = 2*pi*(t-T0)/P'''
    return 1. / (2*np.pi)

def jitter_prior(sigma, sigma0=1):
    sigmaJlim,_,_,_,_,_,_ = _param_conditions()
    sigmamax, sigma0 = float(sigmaJlim[1]), float(sigma0)
    return 1. / (sigma0*(1+sigma/sigma0) * np.log(1+sigmamax/sigma0))

def C_prior(C):
    _,Clim,_,_,_,_,_ = _param_conditions()
    Cmax = float(Clim[1])
    return 1./(2*Cmax) if abs(C) <= Cmax else 0.

def compute_theta_prior(theta):
    theta = np.ascontiguousarray(theta)
    if theta.size == 2:
        sigmaJ,C = theta
        return jitter_prior(sigmaJ) * C_prior(C)

    elif theta.size == 7:
        sigmaJ,C,P1,T01,K1,e1,omega1 = theta
        return jitter_prior(sigmaJ) * C_prior(C) * P_prior(P1) * K_prior(K1) * e_prior() * omega_prior() * M_prior()

    elif theta.size == 12:
        sigmaJ,C,P1,T01,K1,e1,omega1,P2,T02,K2,e2,omega2 = theta
        return jitter_prior(sigmaJ) * C_prior(C) * P_prior(P1) * P_prior(P2) * K_prior(K1) * K_prior(K2) * \
               e_prior()**2 * omega_prior()**2 * M_prior()**2

    elif theta.size == 17:
        sigmaJ,C,P1,T01,K1,e1,omega1,P2,T02,K2,e2,omega2,P3,T03,K3,e3,omega3 = theta
        return jitter_prior(sigmaJ) * C_prior(C) * P_prior(P1) * P_prior(P2) * P_prior(P3) * \
               K_prior(K1) * K_prior(K2) * K_prior(K3) * \
               e_prior()**3 * omega_prior()**3 * M_prior()**3

    else:
        raise ValueError('Weird number of model parameters.')

def planet_prior(Nplanets, alpha=1./3):
    if Nplanets == 0:
        return 1.-np.sum([alpha**i for i in range(1,4)])
    elif 1 <= Nplanets <= 3:
        return alpha**Nplanets
    else:
        return 0.

def compute_planet_prior(theta):
    theta = np.ascontiguousarray(theta)
    if theta.size == 2:
        Nplanets = 0
    elif theta.size == 7:
        Nplanets = 1
    elif theta.size == 12:
        Nplanets = 2
    elif theta.size == 17:
        Nplanets = 3
    else:
        raise ValueError('Weird number of model parameters.')
    return planet_prior(Nplanets)


def precondition_data(theta):
    theta = np.ascontiguousarray(theta)
    if theta.size == 2:
        sigmaJ,C = theta
        return 

    elif theta.size == 7:
        sigmaJ,C,P1,T01,K1,e1,omega1 = theta
        return jitter_prior(sigmaJ) * C_prior(C) * P_prior(P1) * K_prior(K1) * e_prior() * omega_prior() * M_prior()

    elif theta.size == 12:
        sigmaJ,C,P1,T01,K1,e1,omega1,P2,T02,K2,e2,omega2 = theta
        return jitter_prior(sigmaJ) * C_prior(C) * P_prior(P1) * P_prior(P2) * K_prior(K1) * K_prior(K2) * \
               e_prior()**2 * omega_prior()**2 * M_prior()**2

    elif theta.size == 17:
        sigmaJ,C,P1,T01,K1,e1,omega1,P2,T02,K2,e2,omega2,P3,T03,K3,e3,omega3 = theta
        return jitter_prior(sigmaJ) * C_prior(C) * P_prior(P1) * P_prior(P2) * P_prior(P3) * \
               K_prior(K1) * K_prior(K2) * K_prior(K3) * \
               e_prior()**3 * omega_prior()**3 * M_prior()**3

    else:
        raise ValueError('Weird number of model parameters.')
