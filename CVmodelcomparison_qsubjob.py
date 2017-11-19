from imports import *
from scipy.optimize import minimize, fmin_l_bfgs_b
from lnlike import *
from visualize_data import *
from savepickle import *
from priors import *
from run_MCMC import *


def compute_modelposterior_CV(datanum, modelnum, ind, nforecasts, minN_2_fit,
                              factr=1e1, Nmax=2e4):
    '''
    Split the time-series according to ind and compute the lnlikelihood of the 
    rv data given a by the model parameters for the model modelnum. The model 
    should contain between 0-3 planets on keplerian orbits.

    see http://robjhyndman.com/hyndsight/crossvalidation/ 
    Ctrl F: "time series" for splitting info.
    '''
    # Setup
    t0 = time.time()
    folder = 'RVdata%.4d'%datanum
    outsuffix = 'qsubtest_modelnum%i_Ntrain%.3d_nforecasts%i_minN2fit%i'%(modelnum,
									ind, 
									nforecasts, 
									minN_2_fit)
    
    # Get data
    t, rv, erv = get_dataset(datanum)
    sort = np.argsort(t)
    t, rv, erv = t[sort], rv[sort], erv[sort]

    # Split: create training set and a testing point
    if minN_2_fit == 1:
        tokeep = np.delete(np.arange(t.size), ind)
        ttrain, rvtrain, ervtrain = t[tokeep], rv[tokeep], erv[tokeep]
        ttest, rvtest, ervtest = np.ascontiguousarray(t[ind]), \
				 np.ascontiguousarray(rv[ind]), \
				 np.ascontiguousarray(erv[ind])
    else:
    	ttrain, rvtrain, ervtrain = t[:ind], rv[:ind], erv[:ind]
    	ttest, rvtest, ervtest  = np.ascontiguousarray(t[ind+nforecasts-1]), \
                              	  np.ascontiguousarray(rv[ind+nforecasts-1]), \
                              	  np.ascontiguousarray(erv[ind+nforecasts-1])

    # Get initial parameter guesses, bnds, and initial gaussian balls
    theta0_real = get_initializations(datanum, modelnum)
    #bnds = get_bounds(datanum, modelnum)
    initialize = get_gaussianballs(datanum, modelnum)
   
    # Get period limits
    Plims = get_Plims(datanum)
 
    # Optimize keplerian parameters
    sampler, samples, _, results = run_emcee(theta0_real, ttrain, 
					     rvtrain, ervtrain, initialize, 
					     Plims=Plims)
    success = True

    # Find best M (or T0) based on the best-fit P
    if modelnum > 0:
        results = find_optimum_M(results, ttrain, rvtrain, ervtrain)
    theta_real = results[:,0]

    # Get median parameter values
    theta_median = np.median(samples, axis=0)
    
    # Compute prior on the number of planets
    lnmodelpri = np.log(compute_planet_prior(theta_real))

    # Compute lnlikelihood for this training set
    ll = lnlike(theta_real, ttest, rvtest, ervtest)

    # Save results
    try:
	os.mkdir('results')
    except OSError:
	pass
    try:
	os.mkdir('results/%s'%folder)
    except OSError:
	pass
    self = saveRVmodelCV_qsub(time.time()-t0, success, theta0_real, theta_real,
                              initialize, ll, ttrain.size, samples,
                              'results/%s/%s'%(folder, outsuffix))


def get_Plims(datanum):
    Pmins,Pmaxs = np.loadtxt('setup/prior_bounds_%.4d.txt'%datanum, delimiter=',', usecols=(2,3)).T
    return Pmins[0], Pmaxs[0], Pmins[1], Pmaxs[1], Pmins[2], Pmaxs[2]


def MAD(arr):
    return np.median(abs(arr - np.median(arr)))


def find_optimum_M(results, ttrain, rvtrain, ervtrain):
    results_new = results + 0
    Ms = np.linspace(0, 2*np.pi, 100)
    
    if results.shape[0] == 7:
        lls = np.zeros(Ms.size)
        for i in range(Ms.size):
            results_new[3,0] = Ms[i]
            lls[i] = lnlike(results_new[:,0], ttrain, rvtrain, ervtrain)
	if np.any(np.isfinite(lls[i])):
            results_new[3,0] = float(Ms[lls == lls.max()])

    elif results.shape[0] == 12:
        lls = np.zeros((Ms.size, Ms.size))
        for i in range(Ms.size):
            for j in range(Ms.size):
                results_new[3,0], results_new[8,0] = Ms[i], Ms[j]
                lls[i,j] = lnlike(results_new[:,0], ttrain, rvtrain, ervtrain)
	if np.any(np.isfinite(lls[i,j])):
            ind, jind = np.where(lls == lls.max())
            results_new[3,0], results_new[8,0] = float(Ms[ind[0]]), float(Ms[jind[0]])

    elif results.shape[0] == 17:
        lls = np.zeros((Ms.size, Ms.size, Ms.size))
        for i in range(Ms.size):
            for j in range(Ms.size):
                for k in range(Ms.size):
                    results_new[3,0], results_new[8,0], results_new[13,0] = Ms[i], Ms[j], Ms[k]
                    lls[i,j,k] = lnlike(results_new[:,0], ttrain, rvtrain, ervtrain)
	if np.any(np.isfinite(lls[i,j,k])):
            ind, jind, kind = np.where(lls == lls.max())
            results_new[3,0], results_new[8,0], results_new[13,0] = float(Ms[ind[0]]), float(Ms[jind[0]]), \
                                                                    float(Ms[kind[0]])

    else:
        raise ValueError('Weird number of model parameters.')
    
    return results_new


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



if __name__ == '__main__':
    datanum, modelnum, ind, nforecasts, minN_2_fit = int(sys.argv[1]), int(sys.argv[2]), \
                                                     int(sys.argv[3]), int(sys.argv[4]), \
                                                     int(sys.argv[5])
    compute_modelposterior_CV(datanum, modelnum, ind, nforecasts, minN_2_fit)
