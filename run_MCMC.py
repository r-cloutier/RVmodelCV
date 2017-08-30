from imports import *
from visualize_data import *
from priors import compute_theta_prior
from lnlike import lnlike
from scipy.stats import gaussian_kde
from scipy.ndimage.filters import gaussian_filter1d


def get_results_kernel(samples, sigma=10, pltt=True):
    nparams = samples.shape[1]
    results = np.zeros((nparams, 4))
    for i in range(nparams):
        kernel = gaussian_kde(samples[:,i])
        xarr = np.linspace(samples[:,i].min(), samples[:,i].max(), 500)
        probs = kernel.pdf(xarr) / kernel.pdf(xarr).sum()
        probs = gaussian_filter1d(probs, sigma)
        MAP = float(xarr[probs==probs.max()])
        v = np.percentile(samples[:,i], (16,84))
        p1sig, m1sig = v[1]-MAP, MAP-v[0]
        results[i] = [MAP, p1sig, m1sig, np.mean([m1sig,p1sig])]
        if pltt:
            plt.plot(xarr, probs, 'k-')
            plt.axvline(MAP, ls='--')
            plt.axvline(MAP+p1sig, ls='--')
            plt.axvline(MAP-m1sig, ls='--')
            plt.show()
    return results


def lnprob(theta_real, t, rv, erv):
    lp = np.log(compute_theta_prior(theta_real))
    if np.isfinite(lp):
        return lnlike(theta_real, t, rv, erv)
    else:
        return -np.inf

    
def get_initialization(theta):
    #sigmaJ,C,P1,M1,K1,e1,omega1,... = theta
    initialize = np.append([.1,.1], np.tile([1,1,1,1e-2,1],3))
    initialize[12] = 10.
    return initialize[:len(theta)]


def run_emcee(theta_real, t, rv, erv, initialize,
              a=2, nwalkers=100, burnin=200, nsteps=200):
    # Initialize walkers in the parameter space
    ndim, p0 = len(theta_real), []
    #initialize = get_initialization(theta_real)
    for i in range(nwalkers):
        p0.append(theta_real + initialize*np.random.randn(ndim))

    # Initialize sampler
    args = (t, rv, erv)
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
