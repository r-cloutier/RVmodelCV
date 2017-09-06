from imports import *
from savepickle import *


def MAD(arr):
    return np.median(abs(arr - np.median(arr)))
    

def get_median_results(datanum):
    # Get results for this dataset
    pickles = np.array(glob.glob('results/RVdata%.4d/qsub_*'%datanum))
    nmodels = pickles.size

    # Save median lnlikelihoods
    medians, mads, modelnums = np.zeros(nmodels), np.zeros(nmodels), \
                               np.zeros(nmodels),
    for i in range(nmodels):
        self = loadpickle(pickles[i])
        modelnums[i] = int(pickles[i][-1])
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
