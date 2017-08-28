from imports import *
from scipy.optimize import minimize, fmin_l_bfgs_b
from lnlike import *
from visualize_data import *
from savepickle import saveRVmodelCV
from priors import *


def compute_modelposterior_CV(theta, t, rv, erv, bnds, minN_2_fit=20,
                              factr=1e1, Nmax=2e4):
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
    k, theta0s, thetaops = 0, np.zeros((0,len(theta))), np.zeros((0,len(theta)))
    forecaststeps = np.arange(nforecasts)
    for i in range(nforecasts):
        for j in range(T.size):

            # Split: create training set and a testing point
            ind = int(T[j])
            ttrain, rvtrain, ervtrain = t[:ind], rv[:ind], erv[:ind]
            ttest, rvtest, ervtest  = np.ascontiguousarray(t[ind+forecaststeps[i]]), \
                                      np.ascontiguousarray(rv[ind+forecaststeps[i]]), \
                                      np.ascontiguousarray(erv[ind+forecaststeps[i]])

            # Get initial parameter guesses
            theta = thetaops[successes][-1] if True in successes else theta
            theta0s = np.insert(theta0s, k, theta, axis=0)
            
            # Optimize keplerian parameters
            args = (ttrain, rvtrain, ervtrain)
            thetaopt,_,d = fmin_l_bfgs_b(neg_lnlike, x0=theta, args=args, approx_grad=True, 
					 factr=factr, bounds=bnds, maxiter=int(Nmax),
                                         maxfun=int(Nmax))
	    s = True if d['warnflag'] == 0 else False
	    successes = np.append(successes, s)

	    # Save parameter values
	    thetaops = np.insert(thetaops, k, thetaopt, axis=0)
            k += 1
            
            # Compute priors on model parameters
            #lnpri = np.log(compute_theta_prior(thetaopt))

            # Compute prior on the number of planets
            lnmodelpri = np.log(compute_planet_prior(thetaopt))

            # Compute lnlikelihood for this training set
            lnlikes = np.append(lnlikes, lnlike(thetaopt, ttest, rvtest, ervtest) + lnmodelpri)

    # Return mean lnlikelihood and std of the mean
    g = successes
    mad_median = MAD(lnlikes[g]) / np.sqrt(lnlikes[g].size)
    return lnlikes, successes, theta0s, thetaops, np.median(lnlikes[g]), mad_median


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
    bnds = get_bounds(datanum, i)
    t0 = time.time()
    lnlikes, successes, theta0s, thetaops, lls[i], ells[i] = compute_modelposterior_CV(theta, t, rv, erv, bnds)
    successfrac[i] = np.sum(successes) / float(successes.size)
    times[i] = time.time()-t0
    print 'Took %.3e seconds\n'%times[i]

    return times, successfrac, lnlikes, successes, theta0s, thetaops, lls, ells


if __name__ == '__main__':
    nplanets = int(sys.argv[1])
    times, successfrac, lnlikes, successes, theta0s, thetas, lls, ells = run_1model_CV(1, nplanets)
    self = saveRVmodelCV(times, successfrac, lnlikes, successes, theta0s, thetas, lls, ells, 'results/test_wopriors%i'%nplanets)
