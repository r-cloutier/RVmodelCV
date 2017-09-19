from imports import *
from visualize_data import *
from priors import compute_theta_prior, compute_theta_prior_custom
from lnlike import lnlike
from scipy.stats import gaussian_kde
from scipy.ndimage.filters import gaussian_filter1d


def get_results_kernel(samples, sigma=10, pltt=True):
    nparams = samples.shape[1]
    results = np.zeros((nparams, 4))
    Ps_Ks = [2,4,7,9,12,14]
    for i in range(nparams):
	# take log for periods and semiamplitudes
	s = np.log(samples[:,i]) if i in Ps_Ks else samples[:,i]
        s = s[np.isfinite(s)]
	kernel = gaussian_kde(s)
        xarr = np.linspace(s.min(), s.max(), 500)
        probs = kernel.pdf(xarr) / kernel.pdf(xarr).sum()
        probs = gaussian_filter1d(probs, sigma)
        MAP = float(xarr[probs==probs.max()])
        v = np.percentile(s, (16,84))
        p1sig, m1sig = v[1]-MAP, MAP-v[0]
	if i not in Ps_Ks:
            results[i] = [MAP, p1sig, m1sig, np.mean([m1sig,p1sig])]
	else:
	    results[i] = [np.exp(MAP), np.exp(p1sig), np.exp(m1sig), np.mean(np.exp([m1sig,p1sig]))]
        if pltt:
            plt.plot(xarr, probs, 'k-')
            plt.axvline(MAP, ls='--')
            plt.axvline(MAP+p1sig, ls='--')
            plt.axvline(MAP-m1sig, ls='--')
            plt.show()
    return results


def lnprob(theta_real, t, rv, erv, Plims):
    #lp = np.log(compute_theta_prior(theta_real))
    P1min, P1max, P2min, P2max, P3min, P3max = Plims
    lp = np.log(compute_theta_prior_custom(theta_real, P1min, P1max, P2min, P2max, P3min, P3max))
    if np.isfinite(lp):
        return lnlike(theta_real, t, rv, erv) + lp
    else:
        return -np.inf

    
def get_initialization(theta):
    #sigmaJ,C,P1,M1,K1,e1,omega1,... = theta
    initialize = np.append([.1,.1], np.tile([1,1,1,1e-2,1],3))
    initialize[12] = 10.
    return initialize[:len(theta)]


def run_emcee(theta_real, t, rv, erv, initialize, Plims=[]
              a=1.9, nwalkers=100, burnin=400, nsteps=400):
    # Initialize walkers in the parameter space
    ndim, p0 = len(theta_real), []
    #initialize = get_initialization(theta_real)
    for i in range(nwalkers):
        p0.append(theta_real + initialize*np.random.randn(ndim))

    # Initialize sampler
    Plims = Plims if len(list(Plims)) > 0 else (1.25,1e4)*3
    args = (t, rv, erv, Plims)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args, a=a)

    print '\nRunning Burnin...'
    t0 = time.time()
    lnprobs = np.zeros(0)
    p0,_,_ = sampler.run_mcmc(p0, burnin)
    # Save the lnprob to check if the chain converges
    lnprobs = np.append(lnprobs, np.mean(sampler.lnprobability,axis=0))
    print 'Burnin acceptance fraction is %.4f'%np.mean(sampler.acceptance_fraction)
    print 'Burnin took %.4f minutes\n'%((time.time()-t0)/60.)
    sampler.reset()

    print 'Running MCMC (training set)...'
    p0,_,_ = sampler.run_mcmc(p0, nsteps)
    lnprobs = np.append(lnprobs, np.mean(sampler.lnprobability,axis=0))

    print "Mean acceptance fraction: %.4f"%np.mean(sampler.acceptance_fraction)
    print 'Full MCMC took %.4f minutes'%((time.time()-t0)/60.)
    samples = sampler.chain.reshape((-1, ndim))
    return sampler, samples, lnprobs, get_results_kernel(samples, pltt=False)
