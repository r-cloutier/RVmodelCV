from imports import *


def _param_conditions():
    '''
    Return the limits of the parameter values to map input parameters in physical 
    units to the interval [0,1].
    '''
    sigmaJrange = 0., 99.
    Crange = -1e3, 1e3
    Prange = 1.25, 1e4
    Mrange = 0., 2*np.pi
    Krange = 0., 999.
    erange = 0., 1.
    omegarange = 0., 2*np.pi
    return sigmaJrange, Crange, Prange, Mrange, Krange, erange, omegarange


def precondition_data(x, A, B):
    '''
    Phyiscal units to unit interval.
    '''
    A = float(A)
    return -(x - A) / (A - B)


def recondition_data(x, A, B):
    '''
    Unit interval to physical units.
    '''
    A = float(A)
    return -1 * x * (A - B) + A


def precondition_theta(theta_real):
    '''
    Scale parameters in theta to the unit interval.
    '''
    theta_real = np.ascontiguousarray(theta_real)
    sigmaJrange, Crange, Prange, Mrange, Krange, erange, omegarange = _param_conditions()
    if theta_real.size == 2:
        sigmaJ,C = theta_real
        theta_scaled = precondition_data(sigmaJ, sigmaJrange[0], sigmaJrange[1]), \
                       precondition_data(C, Crange[0], Crange[1])

    elif theta_real.size == 7:
        sigmaJ,C,P1,M1,K1,e1,omega1 = theta_real
        theta_scaled =  precondition_data(sigmaJ, sigmaJrange[0], sigmaJrange[1]), \
                        precondition_data(C, Crange[0], Crange[1]), \
                        precondition_data(P1, Prange[0], Prange[1]), \
                        precondition_data(M1, Mrange[0], Mrange[1]), \
                        precondition_data(K1, Krange[0], Krange[1]), \
                        precondition_data(e1, erange[0], erange[1]), \
                        precondition_data(omega1, omegarange[0], omegarange[1])
    
    elif theta_real.size == 12:
        sigmaJ,C,P1,M1,K1,e1,omega1,P2,M2,K2,e2,omega2 = theta_real
        theta_scaled =  precondition_data(sigmaJ, sigmaJrange[0], sigmaJrange[1]), \
                        precondition_data(C, Crange[0], Crange[1]), \
                        precondition_data(P1, Prange[0], Prange[1]), \
                        precondition_data(M1, Mrange[0], Mrange[1]), \
                        precondition_data(K1, Krange[0], Krange[1]), \
                        precondition_data(e1, erange[0], erange[1]), \
                        precondition_data(omega1, omegarange[0], omegarange[1]), \
                        precondition_data(P2, Prange[0], Prange[1]), \
                        precondition_data(M2, Mrange[0], Mrange[1]), \
                        precondition_data(K2, Krange[0], Krange[1]), \
                        precondition_data(e2, erange[0], erange[1]), \
                        precondition_data(omega2, omegarange[0], omegarange[1])
        
    elif theta_real.size == 17:
        sigmaJ,C,P1,M1,K1,e1,omega1,P2,M2,K2,e2,omega2,P3,M3,K3,e3,omega3 = theta_real
        theta_scaled =  precondition_data(sigmaJ, sigmaJrange[0], sigmaJrange[1]), \
                        precondition_data(C, Crange[0], Crange[1]), \
                        precondition_data(P1, Prange[0], Prange[1]), \
                        precondition_data(M1, Mrange[0], Mrange[1]), \
                        precondition_data(K1, Krange[0], Krange[1]), \
                        precondition_data(e1, erange[0], erange[1]), \
                        precondition_data(omega1, omegarange[0], omegarange[1]),\
                        precondition_data(P2, Prange[0], Prange[1]), \
                        precondition_data(M2, Mrange[0], Mrange[1]), \
                        precondition_data(K2, Krange[0], Krange[1]), \
                        precondition_data(e2, erange[0], erange[1]), \
                        precondition_data(omega2, omegarange[0], omegarange[1]), \
                        precondition_data(P3, Prange[0], Prange[1]), \
                        precondition_data(M3, Mrange[0], Mrange[1]), \
                        precondition_data(K3, Krange[0], Krange[1]), \
                        precondition_data(e3, erange[0], erange[1]), \
                        precondition_data(omega3, omegarange[0], omegarange[1])
        
    else:
        raise ValueError('Weird number of model parameters.')

    return np.ascontiguousarray(theta_scaled)


def recondition_theta(theta_scaled):
    '''
    Re-Scale parameters from the unit interval to physical units.
    '''
    theta_scaled = np.ascontiguousarray(theta_scaled)
    sigmaJrange, Crange, Prange, Mrange, Krange, erange, omegarange = _param_conditions()
    if theta_scaled.size == 2:
        sigmaJ,C = theta_scaled
        theta_real = recondition_data(sigmaJ, sigmaJrange[0], sigmaJrange[1]), \
                     recondition_data(C, Crange[0], Crange[1])

    elif theta_scaled.size == 7:
        sigmaJ,C,P1,M1,K1,e1,omega1 = theta_scaled
        theta_real =  recondition_data(sigmaJ, sigmaJrange[0], sigmaJrange[1]), \
                      recondition_data(C, Crange[0], Crange[1]), \
                      recondition_data(P1, Prange[0], Prange[1]), \
                      recondition_data(M1, Mrange[0], Mrange[1]), \
                      recondition_data(K1, Krange[0], Krange[1]), \
                      recondition_data(e1, erange[0], erange[1]), \
                      recondition_data(omega1, omegarange[0], omegarange[1])
    
    elif theta_scaled.size == 12:
        sigmaJ,C,P1,M1,K1,e1,omega1,P2,M2,K2,e2,omega2 = theta_scaled
        theta_real =  recondition_data(sigmaJ, sigmaJrange[0], sigmaJrange[1]), \
                      recondition_data(C, Crange[0], Crange[1]), \
                      recondition_data(P1, Prange[0], Prange[1]), \
                      recondition_data(M1, Mrange[0], Mrange[1]), \
                      recondition_data(K1, Krange[0], Krange[1]), \
                      recondition_data(e1, erange[0], erange[1]), \
                      recondition_data(omega1, omegarange[0], omegarange[1]), \
                      recondition_data(P2, Prange[0], Prange[1]), \
                      recondition_data(M2, Mrange[0], Mrange[1]), \
                      recondition_data(K2, Krange[0], Krange[1]), \
                      recondition_data(e2, erange[0], erange[1]), \
                      recondition_data(omega2, omegarange[0], omegarange[1])
        
    elif theta_scaled.size == 17:
        sigmaJ,C,P1,M1,K1,e1,omega1,P2,M2,K2,e2,omega2,P3,M3,K3,e3,omega3 = theta_scaled
        theta_real =  recondition_data(sigmaJ, sigmaJrange[0], sigmaJrange[1]), \
                      recondition_data(C, Crange[0], Crange[1]), \
                      recondition_data(P1, Prange[0], Prange[1]), \
                      recondition_data(M1, Mrange[0], Mrange[1]), \
                      recondition_data(K1, Krange[0], Krange[1]), \
                      recondition_data(e1, erange[0], erange[1]), \
                      recondition_data(omega1, omegarange[0], omegarange[1]),\
                      recondition_data(P2, Prange[0], Prange[1]), \
                      recondition_data(M2, Mrange[0], Mrange[1]), \
                      recondition_data(K2, Krange[0], Krange[1]), \
                      recondition_data(e2, erange[0], erange[1]), \
                      recondition_data(omega2, omegarange[0], omegarange[1]), \
                      recondition_data(P3, Prange[0], Prange[1]), \
                      recondition_data(M3, Mrange[0], Mrange[1]), \
                      recondition_data(K3, Krange[0], Krange[1]), \
                      recondition_data(e3, erange[0], erange[1]), \
                      recondition_data(omega3, omegarange[0], omegarange[1])
        
    else:
        raise ValueError('Weird number of model parameters.')

    return np.ascontiguousarray(theta_real)


def P_prior(P):
    _,_,Plims,_,_,_,_ = _param_conditions()
    Pmin, Pmax = float(Plims[0]), float(Plims[1])
    return 1. / (P*np.log(Pmax/Pmin)) if Pmin <= P <= Pmax else 0.

def K_prior(K, K0=1):
    _,_,_,_,Klims,_,_ = _param_conditions()
    Kmin, Kmax, K0 = float(Klims[0]), float(Klims[1]), float(K0)
    return 1. / (K0*(1+K/K0) * np.log(1+Kmax/K0)) if Kmin <= K <= Kmax else 0.

def e_prior(e, sigma_e=.2):
    e = float(e)
    PDF_Rayleigh = e/sigma_e**2 * np.exp(-.5*(e/sigma_e)**2)
    CDF_Rayleigh = 1. - np.exp(-.5*(e/sigma_e)**2)
    return PDF_Rayleigh / CDF_Rayleigh if 0 <= e < 1 else 0
        
def omega_prior():
    return 1. / (2*np.pi)

def M_prior():  # mean anomaly
    '''M = n*(t-T0) = 2*pi*(t-T0)/P'''
    return 1. / (2*np.pi)

def jitter_prior(sigma, sigma0=1):
    sigmaJlim,_,_,_,_,_,_ = _param_conditions()
    sigmamin, sigmamax, sigma0 = float(sigmaJlim[0]), float(sigmaJlim[1]), float(sigma0)
    return 1. / (sigma0*(1+sigma/sigma0) * np.log(1+sigmamax/sigma0)) if sigmamin <= sigma <= sigmamax else 0.

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
        sigmaJ,C,P1,M1,K1,e1,omega1 = theta
        return jitter_prior(sigmaJ) * C_prior(C) * P_prior(P1) * K_prior(K1) * e_prior(e1) * omega_prior() * M_prior()

    elif theta.size == 12:
        sigmaJ,C,P1,M1,K1,e1,omega1,P2,M2,K2,e2,omega2 = theta
        return jitter_prior(sigmaJ) * C_prior(C) * P_prior(P1) * P_prior(P2) * K_prior(K1) * K_prior(K2) * \
               e_prior(e1) * e_prior(e2) * omega_prior()**2 * M_prior()**2

    elif theta.size == 17:
        sigmaJ,C,P1,M1,K1,e1,omega1,P2,M2,K2,e2,omega2,P3,M3,K3,e3,omega3 = theta
        return jitter_prior(sigmaJ) * C_prior(C) * P_prior(P1) * P_prior(P2) * P_prior(P3) * \
               K_prior(K1) * K_prior(K2) * K_prior(K3) * \
               e_prior(e1) * e_prior(e2) * e_prior(e3) * omega_prior()**3 * M_prior()**3

    else:
        raise ValueError('Weird number of model parameters.')


def P_prior_custom(P, Pmin, Pmax):
    return 1. / (P*np.log(Pmax/Pmin)) if Pmin <= P <= Pmax else 0.


def compute_theta_prior_custom(theta, P1min, P1max, P2min, P2max, P3min, P3max):
    theta = np.ascontiguousarray(theta)
    if theta.size == 2:
        sigmaJ,C = theta
        return jitter_prior(sigmaJ) * C_prior(C)

    elif theta.size == 7:
        sigmaJ,C,P1,M1,K1,e1,omega1 = theta
        return jitter_prior(sigmaJ) * C_prior(C) * P_prior_custom(P1,P1min,P1max) * K_prior(K1) * \
            e_prior(e1) * omega_prior() * M_prior()

    elif theta.size == 12:
        sigmaJ,C,P1,M1,K1,e1,omega1,P2,M2,K2,e2,omega2 = theta
        return jitter_prior(sigmaJ) * C_prior(C) * P_prior_custom(P1, P1min, P1max) * \
            P_prior_custom(P2, P2min, P2max) * \
            K_prior(K1) * K_prior(K2) * e_prior(e1) * e_prior(e2) * omega_prior()**2 * M_prior()**2

    elif theta.size == 17:
        sigmaJ,C,P1,M1,K1,e1,omega1,P2,M2,K2,e2,omega2,P3,M3,K3,e3,omega3 = theta
        return jitter_prior(sigmaJ) * C_prior(C) * P_prior_custom(P1, P1min, P1max) * \
            P_prior_custom(P2, P2min, P2max) * P_prior_custom(P3, P3min, P3max) * \
               K_prior(K1) * K_prior(K2) * K_prior(K3) * \
               e_prior(e1) * e_prior(e2) * e_prior(e3) * omega_prior()**3 * M_prior()**3

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
