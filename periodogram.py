from imports import *

def compute_LSperiodogram(t, y, ey=np.zeros(0), plims=(.5,2e2)):
    '''Compute the LS periodogram of an input timeseries.'''
    if ey.size == 0:
	ey = np.ones(t.size)
    pmin, pmax = plims
    freqs = 1./np.logspace(np.log10(pmin), np.log10(pmax), 1e3)
    power = LombScargle(t, y, ey).power(freqs)
    powernorm = power / power.std()
    return 1./freqs, power, powernorm


def bootstrap_FAP(t, y, ey, powers=None, plims=(.5,2e2), N=1e4):
    '''Compute the FAP of an input timeseries.'''
    # Get bootstrap estimates of the maximum power
    N, maxpower = int(N), np.zeros(N)
    print '\nComputing bootstrap estimates...'
    t0 = time.time()
    for i in range(N):
        inds = np.arange(y.size)
        np.random.shuffle(inds)
	# bootsrap with replacement: t is fixed but y is shuffled
        _,power,_ = compute_LSperiodogram(t, y[inds], ey[inds], plims=plims)
        maxpower[i] = np.max(power)
    print 'Bootstrapping took %.3f minutes.\n'%((time.time()-t0)/60.)

    # Compute FAP from fraction of bootstraps which exceed a certain power
    if powers == None:
        _,power,_ = compute_LSperiodogram(t, y, ey, plims=plims)
        powers = np.linspace(0, 2*power.max(), 5e3)
    faps = np.zeros(powers.size)
    print 'Computing FAPs from bootstrapping...'
    for i in range(powers.size):
        faps[i] = np.where(maxpower > powers[i])[0].size / float(N)

    # Try and get where FAP = 10, 1, 0.1, .01%
    pow10ind     = abs(faps-1e-1) == abs(faps-1e-1).min()
    fap10, pow10 = faps[pow10ind], powers[pow10ind]
    pow1ind    = abs(faps-1e-2) == abs(faps-1e-2).min()
    fap1, pow1 = faps[pow1ind][-1], powers[pow1ind][-1]
    powd1ind     = abs(faps-1e-3) == abs(faps-1e-3).min()
    fapd1, powd1 = faps[powd1ind][-1], powers[powd1ind][-1]
    powd01ind      = abs(faps-1e-4) == abs(faps-1e-4).min()
    fapd01, powd01 = faps[powd01ind][-1], powers[powd01ind][-1]
    return powers, faps, np.array([pow10, pow1, powd1, powd01]), np.array([fap10, fap1, fapd1, fapd01])
