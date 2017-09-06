from imports import *
from savepickle import *


def MAD(arr):
    return np.median(abs(arr - np.median(arr)))


def get_results(datanum):
    pickles = np.array(glob.glob('results/RVdata%.4d/qsub_*'%datanum))
    pickles = np.sort(pickles)
    selfs = [loadpickle(pickles[i]) for i in range(pickles.size)]
    return selfs


def get_median_results(datanum):
    # Get results for this dataset
    selfs = get_results(datanum)
    nmodels = len(selfs)
    
    # Save median lnlikelihoods
    medians, mads, modelnums = np.zeros(nmodels), np.zeros(nmodels), \
                               np.zeros(nmodels),
    for i in range(nmodels):
        self = selfs[i]
        modelnums[i] = self.model
        medians[i] = np.median(self.lnlikes)
        mads[i] = MAD(self.lnlikes) / np.sqrt(self.lnlikes.size)

    # Sort by model number
    sort = np.argsort(modelnums)
    return modelnums[sort], medians[sort], mads[sort]


def plot_medians(datanum):
    modelnums, medians, mads = get_median_results(datanum)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(modelnums, medians, mads, fmt='k.', capsize=0)
    ax.set_xlabel('Model Number'), ax.set_title('RVdata%.4d'%datanum)
    ax.set_ylabel('Median ln likelihood per measurement')
    ax.set_xlim((-.5,3.5))
    plt.show()
    plt.close('all')


def get_summed_lnlikes(datanum):
    # Get results for this dataset
    selfs = get_results(datanum)
    nmodels = len(selfs)
    
    # Save summed lnlikelihoods
    total_ll, modelnums = np.zeros(nmodels), np.zeros(nmodels)
    for i in range(nmodels):
        self = selfs[i]
        modelnums[i] = self.model
        total_ll[i] = self.lnlikes.sum()

    # Sort by model number
    sort = np.argsort(modelnums)
    return modelnums[sort], total_ll[sort]


def plot_total_lnlikes(datanum):
    modelnums, total_ll = get_summed_lnlikes(datanum)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(modelnums, total_ll, fmt='k.', capsize=0)
    ax.set_xlabel('Model Number'), ax.set_title('RVdata%.4d'%datanum)
    self = loadpickle('results/RVdata%.4d/qsub_datanum%i_modelnum0'%(datanum,datanum))
    ax.set_ylabel('Total ln likelihood for %i measurements'%self.npickles)
    ax.set_xlim((-.5,3.5))
    plt.show()
    plt.close('all')
