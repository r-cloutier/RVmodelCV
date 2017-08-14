from imports import *
from rvmodel import get_rv1
from scipy.optimize import minimize
from lnlike import lnlike


def compute_modelposterior_CV(theta, t, rv, erv, minN_2_fit=20):
    '''
    Do train/test split and compute the lnlikelihood of the rv data given a 
    model described by the model parameters theta. The model should contain 
    between 0-3 planets on keplerian orbits.

    see http://robjhyndman.com/hyndsight/crossvalidation/ 
    Ctrl F: "time series" for splitting info.
    '''
    # Sort data
    sort = np.argsort(t)
    t, rv, erv = t[sort], rv[sort], erv[sort]

    # Define the sizes of the training sets
    nforecasts = 1    # number of steps from last in training set
    T = np.arange(minN_2_fit, bjd.size-nforecasts)
    
    # Loop over each training set and each number of forecast steps
    lnlikes, successes = np.zeros(0), np.zeros(0, dtype=bool)
    forecaststeps = np.arange(nforecasts)
    for i in range(nforecasts):
        for j in range(T.size):

            # Split: create training set and a testing point
            ind = int(T[j])  #t==ind
            ttrain, rvtrain, ervtrain = t[:ind], rv[:ind], erv[:ind]
            ttest, rvtest, ervtest  = t[ind+forecaststeps[i]], \
                                      rv[ind+forecaststeps[i]], \
                                      erv[ind+forecaststeps[i]]
                
            # Optimize keplerian parameters
            nll = lambda *args: -lnlike(*args)
            args = (ttrain, rvtrain, ervtrain)
            result = minimize(nll, theta, args=args)
            thetaopt = result.x if result.success else theta
            successes = np.append(successes, results.success)

            # Compute lnlikelihood for this training set
            lnlikes = np.append(lnlikes, lnlike(thetaopt, ttest,
                                                rvtest, ervtest))

    # Return mean lnlikelihood and std of the mean
    mad_median = MAD(lnlikes) / np.sqrt(lnlikes.size)
    return lnlikes, successes, np.median(lnlikes), mad_median


def MAD(arr):
    return np.median(abs(arr - np.median(arr)))


def preferred_model(models, lls, ells):
    '''Report the favoured model between 2 input lnlikelihoods (and errors) 
    from model_comparison().'''
    assert lls.size == 2
    assert ells.size == 2
    
    # Are the likelihoods conistent within errors
    diff = abs(lls[0]-lls[1])
    sigdiff = np.sqrt(ells[0]**2 + ells[1]**2)
    if diff-sigdiff < 0:
	return models.min()

    # If inconistent then return which model is preferred
    else:
	return models[lls == lls.max()]
