from imports import *
from visualize_data import *


def lnlike(theta, t, rv, erv, GPtheta=(np.sqrt(3),50.,.5,20.)):
    # convert to adnumber objects
    ##theta = adnumber(theta)
    ##t, rv, erv = adnumber(t), adnumber(rv), adnumber(erv)  ## 9 millisec

    # check
    N = t.size
    assert rv.size == N
    assert erv.size == N

    # define fixed covariance hyperparameters
    alpha, le, lp, tau = np.ascontiguousarray(GPtheta)

    # Compute model
    theta = np.ascontiguousarray(theta)
    sigmaJ = theta[0]
    model = compute_rvmodel(theta, t)  ## .06 millisec
    
    # Compute elements of the covariance matrix
    dt = abs(np.tile(t, (N,1)) - np.tile(t, (N,1)).T) ## 1.5 millisec
    K = alpha**2 * np.exp(-.5 * ((np.sin(np.pi*dt/tau) / lp)**2 + (dt/le)**2)) ## 2 millisec
    erv = np.ascontiguousarray(erv).astype(float)
    Sigma = K + np.identity(N) * (erv**2 + sigmaJ**2)  ## 0.3 microsec

    # Invert matrix (using ad.linalg.inv)
    ##Sigmainv = linalg.inv(Sigma)
    Sigmainv = np.linalg.inv(Sigma)    

    # Compute loglikelihood
    ll = -.5 * (np.dot((rv-model).T, np.dot(Sigmainv, (rv-model))) + \
          	np.log(np.linalg.det(Sigma)) + \
                N*np.log(2*np.pi))

    return ll


def neg_lnlike(theta, t, rv, erv, GPtheta=(np.sqrt(3),50.,.5,20.)):
    return -1. * lnlike(theta, t, rv, erv, GPtheta)



# TESTING
if __name__ == '__main__':
    GPtheta=(np.sqrt(3),50.,.5,20.)
    t, rv, erv = get_dataset(1)
    theta = 1., -1.
