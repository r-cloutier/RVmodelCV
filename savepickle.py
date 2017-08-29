from imports import *

class saveRVmodelCV:
    def __init__(self, times_sec, successfrac, lnlikes, successes, theta0s, thetas, lls, ells, outsuffix):
        self.times_sec = times_sec
        self.successfrac = successfrac
	self.lnlikes = lnlikes
	self.successes = successes
	self.thetalabels = ['sigmaJ','C','P1','M1','K1','e1','omega1','P2','M2','K2','e2','omega2','P3','M3','K3','e3','omega3']
	self.theta0s, self.thetas = theta0s, thetas
	self.thetalabels = self.thetalabels[:self.theta0s.shape[1]]
        self.lls, self.ells = lls, ells

        # pickle
	self.outsuffix = outsuffix
        f = open('%s'%self.outsuffix, 'wb')
        pickle.dump(self, f)
        f.close()


class saveRVmodelCV_qsub:
    def __init__(self, time_sec, success, theta0, theta, ll, outsuffix):
	self.time_sec = time_sec
        self.success = success
        self.ll = ll
        self.thetalabels = ['sigmaJ','C','P1','M1','K1','e1','omega1','P2','M2','K2','e2','omega2','P3','M3','K3','e3','omega3']
        self.theta0, self.theta = theta0, theta
        self.thetalabels = self.thetalabels[:self.theta0.size]

        # pickle
	self.outsuffix = outsuffix
        f = open('%s'%self.outsuffix, 'wb')
        pickle.dump(self, f)
        f.close()


def loadpickle(fname):
    f = open(fname, 'rb')
    self = pickle.load(f)
    f.close()
    return self
