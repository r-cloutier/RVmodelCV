from imports import *
from rvmodel import get_rv1
from scipy.optimize import minimize


def lnlike(rvs, model, theta=(np.sqrt(3),50.,.5,20.)):
    

    if x.shape[0] != xp.shape[0] and x.shape[0] != 1:
            raise ValueError('`xp` must have the same number of entries as' + \
                             ' `x` or have only one entry for prediction.')
        Ktmp = np.zeros((x.shape[0], xp.shape[0]))
        for i in xrange(self.Nparams):
            if x.shape[0] == xp.shape[0]:
                xmx = abs(np.tile(x[:,i], (x.shape[0],1)) - \
                          np.tile(xp[:,i], (xp.shape[0],1)).T)
            else:
                xmx = abs(x[:,i].reshape(x.shape[0],1) - xp[:,i])
            Ktmp += np.exp(self.lnhyperparams[i]) * xmx**2
        self.Z = np.exp(self.lnhyperparams[self.Nparams]) * np.exp(-.5 * Ktmp)
        K = np.zeros_like(self.Z) + self.Z
        if x.shape[0] == xp.shape[0]:
            K += np.exp(self.lnhyperparams[self.Nparams+1]) * np.eye(x.shape[0])
        return K
    


#############################################################################
## Define RV models and likelihood
#############################################################################
def lnlike(keptheta, bjd, rv, erv):
    '''Return the likelihood of the model given a set of parameters which 
    uniquely determine the model based on the number of parameters given.'''
    # Compute RV model
    V,P1,T01,K1,h1,k1,P2,T02,K2,h2,k2 = keptheta
    planetmodel1 = get_rv1((P1,T01,0,K1,h1,k1), bjd)
    planetmodel2 = get_rv1((P2,T02,0,K2,h2,k2), bjd)
    model = V+planetmodel1+planetmodel2
    return -.5*(np.sum((rv-model)**2/erv**2 - np.log(1./erv**2)))


def ll_RVmodel1(theta, bjd, rv, erv):
    '''theta = lna,lnl,lnG,lnP,s,V,P1,T01,K1,h1,k1,P2,T02,K2,h2,k2'''
    if theta.size != 16:
        raise ValueError("Theta does not the correct number of parameters.")
    a,l,G,P = np.exp(theta[:4])
    s,V,P1,T01,K1,h1,k1,P2,T02,K2,h2,k2 = theta[4:]
    planetmodel1 = get_rv1((P1,T01,0,K1,h1,k1), bjd)
    planetmodel2 = get_rv1((P2,T02,0,K2,h2,k2), bjd)
    kern1 = george.kernels.ExpSquaredKernel(l)
    kern2 = george.kernels.ExpSine2Kernel(G,P)
    gp = george.GP(a*(kern1+kern2))
    try:
        gp.compute(bjd, np.sqrt(erv**2 + s**2))
    except (ValueError, np.linalg.LinAlgError):
        return -np.inf
    return gp.lnlikelihood(rv-V-planetmodel1-planetmodel2, quiet=True)


def ll_RVmodel2(theta, bjd, rv, erv):
    '''theta = lna,lnl,lnG,lnP,s,V,P1,T01,K1,h1,k1,P2,T02,K2,h2,k2'''
    if theta.size != 16:
        raise ValueError("Theta does not the correct number of parameters.")
    a,l,G,P = np.exp(theta[:4])
    s,V,P1,T01,K1,h1,k1,P2,T02,K2,h2,k2 = theta[4:]
    planetmodel1 = get_rv1((P1,T01,0,K1,h1,k1), bjd)
    planetmodel2 = get_rv1((P2,T02,0,K2,h2,k2), bjd)
    kern1 = george.kernels.ExpSquaredKernel(l)
    kern2 = george.kernels.ExpSine2Kernel(G,P)
    gp = george.GP(a*(kern1+kern2))
    try:
        gp.compute(bjd, np.sqrt(erv**2 + s**2))
    except (ValueError, np.linalg.LinAlgError):
        return -np.inf
    return gp.lnlikelihood(rv-V-planetmodel1-planetmodel2, quiet=True)

def ll_RVmodel4(theta, bjd, rv, erv):
    '''theta = lna,lnl,lnG,lnP,s,V,P1,T01,K1,h1,k1,P2,T02,K2,h2,k2'''
    if theta.size != 16:
        raise ValueError("Theta does not the correct number of parameters.")
    a,l,G,P = np.exp(theta[:4])
    s,V,P1,T01,K1,h1,k1,P2,T02,K2,h2,k2 = theta[4:]
    planetmodel1 = get_rv1((P1,T01,0,K1,h1,k1), bjd)
    planetmodel2 = get_rv1((P2,T02,0,K2,h2,k2), bjd)
    kern1 = george.kernels.ExpSquaredKernel(l)
    kern2 = george.kernels.ExpSine2Kernel(G,P)
    gp = george.GP(a*(kern1+kern2))
    try:
        gp.compute(bjd, np.sqrt(erv**2 + s**2))
    except (ValueError, np.linalg.LinAlgError):
        return -np.inf
    return gp.lnlikelihood(rv-V-planetmodel1-planetmodel2, quiet=True)


def ll_RVmodel3(theta, bjd, rv, erv):
    '''theta = V,P1,T01,K1,h1,k1,P2,T02,K2,h2,k2'''
    if theta.size != 11:
        raise ValueError("Theta does not the correct number of parameters.")
    V,P1,T01,K1,h1,k1,P2,T02,K2,h2,k2 = theta
    planetmodel1 = get_rv1((P1,T01,0,K1,h1,k1), bjd)
    planetmodel2 = get_rv1((P2,T02,0,K2,h2,k2), bjd)
    kern1 = george.kernels.ExpSquaredKernel(1)
    kern2 = george.kernels.ExpSine2Kernel(1,1)
    gp = george.GP(0*(kern1+kern2))
    try:
        gp.compute(bjd, erv)
    except (ValueError, np.linalg.LinAlgError):
        return -np.inf
    return gp.lnlikelihood(rv-V-planetmodel1-planetmodel2, quiet=True)


#############################################################################
## Timeseries CV
#############################################################################
def compute_lL_CV(theta, model, bjd, rv, erv, minN_2_fit=20):
    '''Do timeseries split and compute the mean lnlikelihood of the model for 
    each fold. Return the average lnlikelihood for this model. The number of 
    parameters in theta defines the model.
    see http://robjhyndman.com/hyndsight/crossvalidation/ Ctrl F: "time series" 
    for splitting info.'''
    # What model are we dealing with (1, 2, or 3)?
    if model == 1:
        a,l,G,P = np.exp(theta[:4])
        s,V,P1,T01,K1,h1,k1,P2,T02,K2,h2,k2 = theta[4:]
        llRVfunc = ll_RVmodel1
    elif model == 2:
        a,l,G,P = np.exp(theta[:4])
        s,V,P1,T01,K1,h1,k1,P2,T02,K2,h2,k2 = theta[4:]
        llRVfunc = ll_RVmodel2
    elif model == 3:
        a,l,G,P,s = 0,1,1,1,0
        V,P1,T01,K1,h1,k1,P2,T02,K2,h2,k2 = theta
        llRVfunc = ll_RVmodel3
    elif model == 4:
	a,l,G,P = np.exp(theta[:4])
        s,V,P1,T01,K1,h1,k1,P2,T02,K2,h2,k2 = theta[4:]
        llRVfunc = ll_RVmodel4
    else:
        raise ValueError("Unsure of what model this is.")

    # Sort data chronologically
    sort = np.argsort(bjd)
    bjd, rv, erv = bjd[sort], rv[sort], erv[sort]

    # Define the sizes of the training sets
    nforecasts = 3    # number of steps from last in training set
    T = np.arange(minN_2_fit, bjd.size-nforecasts)
    
    # Loop over each training set
    lnlikes = np.zeros(0)
    forecaststeps = np.arange(nforecasts)
    for i in range(forecaststeps.size):
        for j in range(T.size):
            # Split: create training set and testing point
            t = int(T[j])
            bjdtrain, rvtrain, ervtrain = bjd[:t], rv[:t], erv[:t]
            bjdtest, rvtest, ervtest  = bjd[t+forecaststeps[i]], \
                                        rv[t+forecaststeps[i]], \
                                        erv[t+forecaststeps[i]]

            # Fit GP model to the training set
            if model in [1,2,4]:
                planetmodel1 = get_rv1((P1,T01,0,K1,h1,k1), bjdtrain)
                planetmodel2 = get_rv1((P2,T02,0,K2,h2,k2), bjdtrain)
                kern1 = george.kernels.ExpSquaredKernel(l)
                kern2 = george.kernels.ExpSine2Kernel(G,P)
                gp = george.GP(a*(kern1+kern2))
                try:
                    gp.compute(bjdtrain, np.sqrt(ervtrain**2 + s**2))
                except (ValueError, np.linalg.LinAlgError):
                    return -np.inf
                gp.optimize(bjdtrain, rvtrain-V-planetmodel1-planetmodel2,
                            np.sqrt(ervtrain**2 + s**2), verbose=False)

                # Get best-fit GP model
                a,l,G,P = np.exp(gp.kernel.vector)
                kern1 = george.kernels.ExpSquaredKernel(l)
                kern2 = george.kernels.ExpSine2Kernel(G,P)
                gp = george.GP(a*(kern1+kern2))
                gp.compute(bjdtrain, np.sqrt(ervtrain**2 + s**2))
                mutrain = gp.predict(rvtrain-V-planetmodel1-planetmodel2,
                                     bjdtrain, mean_only=True)
            else:
                mutrain = np.zeros(bjdtrain.size)    
                
            # Optimize keplerian parameters
            kepthetatmp = np.array([V,P1,T01,K1,h1,k1,P2,T02,K2,h2,k2])
            nll = lambda *args: -lnlike(*args)
            args=(bjdtrain, rvtrain-mutrain, ervtrain)
            result = minimize(nll, kepthetatmp, args=args)
            kepthetaopt = result.x if result.success else kepthetatmp
            
            # Compute lnlikelihood for this training set
            if model in [1,2,4]:
                V,P1,T01,K1,h1,k1,P2,T02,K2,h2,k2 = kepthetaopt
                planetmodel1 = get_rv1((P1,T01,0,K1,h1,k1), [bjdtest])
                planetmodel2 = get_rv1((P2,T02,0,K2,h2,k2), [bjdtest])
                kern1 = george.kernels.ExpSquaredKernel(l)
                kern2 = george.kernels.ExpSine2Kernel(G,P)
                gp = george.GP(a*(kern1+kern2))
                try:
                    gp.compute(bjdtest, np.sqrt(ervtest**2 + s**2))
                except (ValueError, np.linalg.LinAlgError):
                    return -np.inf
                mutest = gp.predict(rvtest-V-planetmodel1-planetmodel2,
                                    bjdtest, mean_only=True)
            else:
                mutest = np.zeros(rvtest.size)
            lnlikes = np.append(lnlikes, lnlike(kepthetaopt,
                                                np.array([bjdtest]),
                                                rvtest-mutest, ervtest))
            
    # Return mean lnlikelihood and std of the mean
    mad_median = MAD(lnlikes) / np.sqrt(lnlikes.size)
    return lnlikes, np.median(lnlikes), mad_median


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
    theta1 = np.array([ -4.66982736e-01,   4.16650158e+00,   1.91462647e-01,
                        3.65358796e+00,   2.58914616e-01,   6.53748798e+02,
                        8.96400823e+00,   4.32832725e+00,   4.68738832e+00,
                        -3.73064515e-01,   4.65244549e-02,   3.29396265e+01,
                        2.94488104e-03,   3.18148982e+00,   3.34980779e-01,
                        -1.55283646e-01])
    theta2 = np.array([  2.30438669e-01,   1.63161824e+01,  -9.78287255e-04,
                         3.69938771e+00,   5.51801052e-01,   6.53697531e+02,
                         8.96579703e+00,   2.45726455e+06,   4.74366537e+00,
                         -4.60342568e-01,  -4.11355586e-02,   3.29396238e+01,
                         2.45726439e+06,   3.31826578e+00,   3.27642681e-01,
                         -1.57454936e-01])
    theta3 = np.array([  6.53658611e+02,   8.96598533e+00,   2.45726448e+06,
                         4.62725767e+00,  -5.09642935e-01,  -6.54260919e-03,
                         3.29396008e+01,   2.45726439e+06,   3.25136375e+00,
                         3.19323454e-01,  -1.06312033e-01])
    theta4 = np.array([  1.35279427e+00,   3.46846096e+00,   1.52137165e-01,
         		 3.60840934e+00,   2.20816144e-01,   6.53716473e+02,
         		 8.96575772e+00,   2.45726449e+06,   4.52321055e+00,
        		 -3.71311641e-01,  -4.45712145e-03,   3.29396246e+01,
         		 2.45726439e+06,   3.25646919e+00,   3.12268093e-01,
        		 -1.88759412e-01])

    bjd, rv, erv, Sindex, Halpha, fwhm, bis = readdata_K218()
    t0 = time.time()
    models, lls, ells = np.arange(4)+1, np.zeros(4), np.zeros(4)

    print 'Model 1...'
    lnlikes1, lls[0], ells[0] = compute_lL_CV(theta1, 1, bjd, rv, erv)

    print 'Model 2...'
    lnlikes2, lls[1], ells[1] = compute_lL_CV(theta2, 2, bjd, rv, erv)

    print 'Model 3...'
    lnlikes3, lls[2], ells[2] = compute_lL_CV(theta3, 3, bjd, rv, erv)

    print 'Model 4...'
    lnlikes4, lls[3], ells[3] = compute_lL_CV(theta4, 4, bjd, rv, erv)
    print 'CV took %.3f minutes'%((time.time()-t0)/60.)
