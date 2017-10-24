'''
Add priors and increase evidences to the full dataset size.
'''
from imports import *
from priors import *


def get_lls(datanum, iteration):
    '''
    datanum = int in [1,6]
    iteration = string in ['a','b']
    '''
    assert datanum in range(1,7)
    assert iteration in ['a','b']
    fname = 'Cloutier/TimeSeriesCV/evidences_%.4d%s.txt'%(datanum, iteration)
    lls, ells = np.loadtxt(fname, usecols=(3,5), delimiter=',').T
    return lls, ells


def get_parameters(datanum, modelnum, iteration):
    '''
    datanum = int in [1,6]
    modelnum = int in [0,3]
    iteration = string in ['a','b']
    '''
    assert datanum in range(1,7)
    assert modelnum in range(4)
    assert iteration in ['a','b']
    fname = 'Cloutier/TimeSeriesCV/params_%i_%.4d%s.txt'%(modelnum,
                                                          datanum, iteration)
    vals, sigs = np.loadtxt(fname, usecols=(4,5), delimiter=',').T
    return vals, sigs


def get_prior(datanum, modelnum, iteration):
    '''
    datanum = int in [1,6]
    modelnum = int in [0,3]
    iteration = string in ['a','b']
    '''
    vals,_ = get_parameters(datanum, modelnum, iteration)
    if modelnum > 0:
        vals = np.append(vals, np.nan)
    lnmodelpri = np.log(compute_planet_prior(vals))

    return lnmodelpri
    
    
def scale_ll(datanum, iteration, Nobs=2e2):
    # Get median lnlikelihoods
    medlls, emedlls = get_lls(datanum, iteration)

    # Scale lnlikelihoods
    lls, ells = np.zeros(medlls.size), np.zeros(medlls.size)
    for modelnum in range(medlls.size):
        print datanum, modelnum, iteration
        lp = get_prior(datanum, modelnum, iteration)
        print lp
        lls[i] = medlls*Nobs + lp
        ells[i] = emedlls*Nobs + lp
        
    return lls, ells
    
