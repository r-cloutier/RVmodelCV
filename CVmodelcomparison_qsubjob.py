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
    ttrain, rvtrain, ervtrain = t[:ind], rv[:ind], erv[:ind]
    ttest, rvtest, ervtest  = np.ascontiguousarray(t[ind+nforecasts-1]), \
                              np.ascontiguousarray(rv[ind+nforecasts-1]), \
                              np.ascontiguousarray(erv[ind+nforecasts-1])

    # Get initial parameter guesses, bnds, and initial gaussian balls
    theta0_real = get_initializations(datanum, modelnum)
    #bnds = get_bounds(datanum, modelnum)
    initialize = get_gaussianballs(datanum, modelnum)
    
    # Optimize keplerian parameters
    #args = (ttrain, rvtrain, ervtrain)
    #theta_real,_,d = fmin_l_bfgs_b(neg_lnlike, x0=theta0_real, args=args,
    #                               approx_grad=True, factr=factr, bounds=bnds,
    #                               maxiter=int(Nmax), maxfun=int(Nmax))
    #success = True if d['warnflag'] == 0 else False
    sampler, samples, _, results = run_emcee(theta0_real, ttrain, 
					     rvtrain, ervtrain, initialize)
    success = True
    theta_real = results[:,0]
 
    # Compute prior on the number of planets
    lnmodelpri = np.log(compute_planet_prior(theta_real))

    # Compute lnlikelihood for this training set
    ll = lnlike(theta_real, ttest, rvtest, ervtest)# + lnmodelpri

    # Save results
    try:
	os.mkdir('results')
    except OSError:
	pass
    try:
	os.mkdir('results/%s'%folder)
    except OSError:
	pass
    self = saveRVmodelCV_qsub(time.time()-t0, success, theta0_real, theta_real, ll, ttrain.size,
                              samples, 'results/%s/%s'%(folder, outsuffix))


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



if __name__ == '__main__':
    datanum, modelnum, ind, nforecasts, minN_2_fit = int(sys.argv[1]), int(sys.argv[2]), \
                                                     int(sys.argv[3]), int(sys.argv[4]), \
                                                     int(sys.argv[5])
    compute_modelposterior_CV(datanum, modelnum, ind, nforecasts, minN_2_fit)
