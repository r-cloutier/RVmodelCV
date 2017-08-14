from imports import *
from periodogram import compute_LSperiodogram


def get_dataset(num):
    assert 1 <= num <= 6
    t, rv, erv = np.loadtxt('Inputs/data/rvs_%.4d.txt'%num).T
    return t, rv, erv


def plot_periodogram(t, rv, erv, plims=(1,1e4)):
    per, pwr, pwrn = compute_LSperiodogram(t, rv, erv, plims=plims)
    # plotting
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    ax.plot(per, pwrn, 'k-')
    ax.set_xscale('log')
    ax.set_xlabel('Period (days)'), ax.set_ylabel('Normalized Power')
    plt.show()
    plt.close('all')

def plot_phasedRVs(t, rv, erv, P, T0=0):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    phase = foldAt(t, P, T0=T0)
    ax.errorbar(phase, rv, erv, fmt='k.', capsize=0, elinewidth=.9)
    ax.set_xlabel('Phase (%.3f days)'%P), ax.set_ylabel('RV (m/s)')
    plt.show()
    plt.close('all')
