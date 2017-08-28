from imports import *

class saveRVmodelCV:
    def __init__(self, times_sec, successfrac, lnlikes, successes, theta0s, thetas, lls, ells, outsuffix):
        self.times_sec = times_sec
        self.successfrac = successfrac
	self.lnlikes = lnlikes
	self.successes = successes
	self.theta0s, self.thetas = theta0s, thetas
        self.lls, self.ells = lls, ells

        # pickle
        f = open('%s'%outsuffix, 'wb')
        pickle.dump(self, f)
        f.close()

def loadpickle(fname):
    f = open(fname, 'rb')
    self = pickle.load(f)
    f.close()
    return self
