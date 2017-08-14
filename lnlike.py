from imports import *
from visualize_data import compute_rvmodel


def lnlike(theta, t, rvs, erv, GPtheta=(np.sqrt(3),50.,.5,20.)):
    # check
    N = t.size
    assert rvs.size == N
    assert erv.size == N

    # define fixed covariance hyperparameters
    alpha, le, lp, tau = GPtheta

    # Compute model
    theta = np.ascontiguousarray(theta)
    sigmaJ = theta[0]
    model = compute_rvmodel(theta, t)
    
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
