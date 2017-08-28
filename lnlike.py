from imports import *
from visualize_data import *
from priors import recondition_theta

def lnlike(theta_scaled, t, rv, erv, GPtheta=(np.sqrt(3),50.,.5,20.)):
    # check
    N = t.size
    assert rv.size == N
    assert erv.size == N

    # define fixed covariance hyperparameters
    alpha, le, lp, tau = np.ascontiguousarray(GPtheta)

    # Compute model
    theta_scaled = np.ascontiguousarray(theta_scaled)
    theta_real = recondition_data(theta_scaled)
    sigmaJ = theta_real[0]
    model = compute_rvmodel(theta_real, t)
    
    # Compute elements of the covariance matrix
    dt = abs(np.tile(t, (N,1)) - np.tile(t, (N,1)).T)
    K = alpha**2 * np.exp(-.5 * ((np.sin(np.pi*dt/tau) / lp)**2 + (dt/le)**2))
    erv = np.ascontiguousarray(erv).astype(float)
    Sigma = K + np.identity(N) * (erv**2 + sigmaJ**2)

    # Invert matrix
    Sigmainv = np.linalg.inv(Sigma)    

    # Compute loglikelihood
    ll = -.5 * (np.dot((rv-model).T, np.dot(Sigmainv, (rv-model))) + \
          	np.log(np.linalg.det(Sigma)) + \
                N*np.log(2*np.pi))

    return ll


def neg_lnlike(theta_scaled, t, rv, erv, GPtheta=(np.sqrt(3),50.,.5,20.)):
    return -1. * lnlike(theta_scaled, t, rv, erv, GPtheta)
