from imports import *
from rvmodel import get_rv2


def get_h_k(e, omega):
    return np.sqrt(e)*np.cos(omega), np.sqrt(e)*np.sin(omega)


def lnlike(theta, t, rvs, erv, GPtheta=(np.sqrt(3),50.,.5,20.)):
    # check
    N = t.size
    assert rvs.size == N
    assert model.size == N

    # define fixed covariance hyperparameters
    alpha, le, lp, tau = GPtheta

    # Compute planet models
    model = np.zeros(N)
    if theta.size == 7:
        sigmaJ,C,P1,T01,K1,e1,omega1 = theta
        h1, k1 = get_h_K(e1, omega1)
        model += get_rv2((P1,T01,K1,h1,k1), t)
        
    elif theta.size == 12:
        sigmaJ,C,P1,T01,K1,e1,omega1,P2,T02,K2,e2,omega2 = theta
        h1, k1 = get_h_K(e1, omega1)
        h2, k2 = get_h_K(e2, omega2)
        model += get_rv2((P1,T01,K1,h1,k1), t)
        model += get_rv2((P2,T02,K2,h2,k2), t)
        
    elif theta.size == 17:
        sigmaJ,C,P1,T01,K1,e1,omega1,P2,T02,K2,e2,omega2,P3,T03,K3,e3,omega3 = theta
        h1, k1 = get_h_K(e1, omega1)
        h2, k2 = get_h_K(e2, omega2)
        h3, k3 = get_h_K(e3, omega3)
        model += get_rv2((P1,T01,K1,h1,k1), t)
        model += get_rv2((P2,T02,K2,h2,k2), t)
        model += get_rv2((P3,T03,K3,h3,k3), t)

    else:
        raise ValueError('Weird number of model parameters.')

    # Add systemic velocity
    model += C
    
    # Compute elements of the covariance matrix
    dt = abs(np.tile(t, (N,1)) - np.tile(t, (N,1)).T)
    K = alpha**2 * np.exp(-.5 * ((np.sin(np.pi*dt/tau) / lp)**2 + (dt/le)**2))
    Sigma = K + np.identity(N) * (erv**2 + sigmaJ**2)

    # Invert matrix
    Sigmainv = np.linalg.inv(Sigma)
    
    # Compute loglikelihood
    return -.5 * (np.dot((rvs-model).T, np.dot(Sigmainv, (rvs-model))) + \
                  np.log(np.linalg.det(Sigma)) + \
                  N*np.log(2*np.pi))
