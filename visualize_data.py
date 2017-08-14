from imports import *
from periodogram import compute_LSperiodogram
from rvmodel import get_rv2


def get_dataset(num):
    assert 1 <= num <= 6
    t, rv, erv = np.loadtxt('Inputs/data/rvs_%.4d.txt'%num).T
    return t, rv, erv

def get_h_k(e, omega):
    return np.sqrt(e)*np.cos(omega), np.sqrt(e)*np.sin(omega)


def compute_rvmodel(theta, t):
    N, theta = t.size, np.ascontiguousarray(theta)
    model = np.zeros(N)
    if theta.size == 2:
	sigmaJ,C = theta
        model = np.repeat(C, N)

    elif theta.size == 7:
        sigmaJ,C,P1,T01,K1,e1,omega1 = theta
        h1, k1 = get_h_k(e1, omega1)
        model += get_rv2((P1,T01,K1,h1,k1), t)

    elif theta.size == 12:
        sigmaJ,C,P1,T01,K1,e1,omega1,P2,T02,K2,e2,omega2 = theta
        h1, k1 = get_h_k(e1, omega1)
        h2, k2 = get_h_k(e2, omega2)
        model += get_rv2((P1,T01,K1,h1,k1), t)
        model += get_rv2((P2,T02,K2,h2,k2), t)

    elif theta.size == 17:
        sigmaJ,C,P1,T01,K1,e1,omega1,P2,T02,K2,e2,omega2,P3,T03,K3,e3,omega3 = theta
        h1, k1 = get_h_k(e1, omega1)
        h2, k2 = get_h_k(e2, omega2)
        h3, k3 = get_h_k(e3, omega3)
        model += get_rv2((P1,T01,K1,h1,k1), t)
        model += get_rv2((P2,T02,K2,h2,k2), t)
        model += get_rv2((P3,T03,K3,h3,k3), t)

    else:
        raise ValueError('Weird number of model parameters.')

    return model+C


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


def plot_phasedRVs(t, rv, erv, P, T0=0, model=np.zeros(0)):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    phase = foldAt(t, P, T0=T0)
    ax.errorbar(phase, rv, erv, fmt='k.', capsize=0, elinewidth=.9)
    if model.size == t.size:
	sort = np.argsort(phase)
	ax.plot(phase[sort], model[sort], 'b-')
    ax.set_xlabel('Phase (%.3f days)'%P), ax.set_ylabel('RV (m/s)')
    plt.show()
    plt.close('all')
