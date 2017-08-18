from imports import *
from scipy.optimize import minimize
from lnlike import lnlike
from visualize_data import *
from savepickle import saveRVmodelCV
from priors import *


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
    T = np.arange(minN_2_fit, t.size-nforecasts)
    
    # Loop over each training set and each number of forecast steps
    lnlikes, successes = np.zeros(0), np.zeros(0, dtype=bool)
    k, thetaops = 0, np.zeros((0,len(theta)))
    forecaststeps = np.arange(nforecasts)
    for i in range(nforecasts):
        for j in range(T.size):

            # Split: create training set and a testing point
            ind = int(T[j])
            ttrain, rvtrain, ervtrain = t[:ind], rv[:ind], erv[:ind]
            ttest, rvtest, ervtest  = np.ascontiguousarray(t[ind+forecaststeps[i]]), \
                                      np.ascontiguousarray(rv[ind+forecaststeps[i]]), \
                                      np.ascontiguousarray(erv[ind+forecaststeps[i]])

            # Optimize keplerian parameters
            nll = lambda *args: -lnlike(*args)
            args = (ttrain, rvtrain, ervtrain)
            result = minimize(nll, theta, args=args)
            thetaopt = result.x if result.success else theta
            successes = np.append(successes, result.success)
            #thetaopt = theta
            #successes = np.append(successes, True)

	    # Save parameter values
	    thetaops = np.insert(thetaops, k, thetaopt, axis=0)

            # Compute priors on model parameters
            #lnpri = np.log(compute_theta_prior(thetaopt))

            # Compute prior on the number of planets
            #lnmodelpri = np.log(compute_planet_prior(thetaopt))

            # Compute lnlikelihood for this training set
            lnlikes = np.append(lnlikes, lnlike(thetaopt, ttest, rvtest, ervtest)) #+ \
                                         #lnpri + lnmodelpri)

    # Return mean lnlikelihood and std of the mean
    mad_median = MAD(lnlikes) / np.sqrt(lnlikes.size)
    return lnlikes, successes, thetaops, np.median(lnlikes), mad_median


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


def run_1model_CV(datanum, modelnum):
    # Get data and theta
    t, rv, erv = get_dataset(datanum)

    # Run CV on each planet model
    times, successfrac, lls, ells = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
    i = modelnum
    print 'CV on %i planet model...'%i
    theta = get_initializations(datanum, i)
    t0 = time.time()
    lnlikes, successes, thetaops, lls[i], ells[i] = compute_modelposterior_CV(theta, t, rv, erv)
    successfrac[i] = successes.sum() / float(successes.size)
    times[i] = time.time()-t0
    print 'Took %.3e seconds\n'%times[i]
   
    return times, successfrac, lnlikes, successes, thetaops, lls, ells


if __name__ == '__main__':
    nplanets = int(sys.argv[1])
    times, successfrac, lnlikes, successes, thetas, lls, ells = run_1model_CV(1, nplanets)
    self = saveRVmodelCV(times, successfrac, lnlikes, successes, lls, ells, 'results/test_wpriors%i'%nplanets)
