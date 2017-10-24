'''
Add priors and increase evidences to the full dataset size.
'''
from imports import *
from priors import *
from uncertainties import unumpy as unp


def get_e_omega(h, k):
    return h*h + k*k, np.arctan2(k, h)


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
    theta,_ = get_parameters(datanum, modelnum, iteration)
    if modelnum == 0 :
        C, J = theta
        theta = J, C
        
    elif modelnum == 1:
        P1, K1, h1, k1, C, J = theta
        e1, omega1 = get_e_omega(h1, k1)
        theta = J, C, P1, 0, K1, e1, omega1 

    elif modelnum == 2:
        P1, K1, h1, k1, P2, K2, h2, k2, C, J = theta
        e1, omega1 = get_e_omega(h1, k1)
        e2, omega2 = get_e_omega(h2, k2)
        theta = J, C, P1, 0, K1, e1, omega1, P2, 0, K2, e2, omega2
        
    else:
        P1, K1, h1, k1, P2, K2, h2, k2, P3, K3, h3, k3, C, J = theta
        e1, omega1 = get_e_omega(h1, k1)
        e2, omega2 = get_e_omega(h2, k2)
        e3, omega3 = get_e_omega(h3, k3)
        theta = J, C, P1, 0, K1, e1, omega1, P2, 0, K2, e2, omega2, \
                P3, 0, K3, e3, omega3

    lp = np.log(compute_theta_prior(theta))
    lnmodelpri = np.log(compute_planet_prior(np.zeros(modelnum*5+2)))
    return lp+lnmodelpri
    
    
def scale_ll(datanum, iteration, Nobs=2e2):
    # Get median lnlikelihoods
    medlls, emedlls = get_lls(datanum, iteration)

    # Scale lnlikelihoods
    lls, ells = np.zeros(medlls.size), np.zeros(medlls.size)
    for modelnum in range(medlls.size):
        #lp = get_prior(datanum, modelnum, iteration)
        i = modelnum
        lls[i] = medlls[i]*Nobs# + lp
        ells[i] = emedlls[i]*Nobs# + lp
        
    return lls, ells
    

def get_oddsratio_file(datanum, iteration):
    lls, ells = scale_ll(datanum, iteration)
    lls = unp.uarray(lls, ells)

    # compute odd ratios
    numerator, denominator = np.zeros(3), np.zeros(3)
    median_ratio, sigma_ratio = np.zeros(3), np.zeros(3)
    for i in range(3):
        numerator[i], denominator[i] = i+1, i
        log10_oddsratio = unp.log10(unp.exp(lls[i+1]-lls[i]))    
        median_ratio[i], sigma_ratio[i] = unp.nominal_values(log10_oddsratio), \
                                          unp.std_devs(log10_oddsratio)
                                          
    # create odds ratio files
    f = open('Cloutier/TimeSeriesCV/oddsratio_%.4d%s.txt'%(datanum, iteration),
             'w')
    g = '# Number of lnL evaluations, planet model denominator, planet model numerator, mode log10(odds ratio), median log10(odds ratio), minus 2 sigma (not used), minus 1 sigma, plus 1 sigma, plus 2 sigma (not used)'
    for i in range(3):
        g += '\n1.6E+07,%i,%i,%.6f,%.6f,NaN,%.6f,%.6f,NaN'%(denominator[i],
                                                            numerator[i],
                                                            median_ratio[i],
                                                            median_ratio[i],
                                                            sigma_ratio[i],
                                                            sigma_ratio[i])
    f.write(g)
    f.close()
