from imports import *

class saveRVmodelCV:
    def __init__(self, times, successfrac, lls, ells, outsuffix):
        self.times = times
        self.successfrac = successfrac
        self.lls, self.ells = lls, ells

        # pickle
        f = open('results_%s'%outsuffix, 'wb')
        pickle.dump(self, f)
        f.close()

def loadpickle(fname):
    f = open(fname, 'rb')
    self = pickle.load(f)
    f.close()
    return self
