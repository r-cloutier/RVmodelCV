from imports import *
from periodogram import compute_LSperiodogram
from rvmodel import get_rv2


def get_dataset(num):
    assert 1 <= num <= 6
    t, rv, erv = np.loadtxt('data/rvs_%.4d.txt'%num).T
    return t, rv, erv


def get_initializations(num, nplanets):
    assert 1 <= num <= 6
    assert 0 <= nplanets <= 3
    f = open('setup/initializations_%.4d.dat'%num,'r')
    g = f.readlines()
    f.close()
    theta_real = [float(i) for i in g[nplanets].split(',')]
    return np.ascontiguousarray(theta_real)


def get_bounds(num, nplanets):
    assert 1 <= num <= 6
    assert 0 <= nplanets <= 3
    f = open('setup/bounds_%.4d.dat'%num,'r')
    g = f.readlines()
    f.close()
    bnds = []
    for i in range(len(g)):
	ls = g[i].split(',')
	assert len(ls) == 2
	try:
	    bnd0 = float(ls[0])
	except ValueError:
	    bnd0 = None
        try:
            bnd1 = float(ls[1])
        except ValueError:
            bnd1 = None
	bnds.append((bnd0, bnd1))
    return bnds[:2+int(nplanets)*5]


def get_h_k(e, omega):
    return np.sqrt(e)*np.cos(omega), np.sqrt(e)*np.sin(omega)


def get_T0(P, M):
    return P * M/(2*np.pi)


def compute_rvmodel(theta_real, t):
    N, theta_real = t.size, np.ascontiguousarray(theta_real)
    model = np.zeros(N)
    if theta_real.size == 2:
	sigmaJ,C = theta_real
        model = np.zeros(N)

    elif theta_real.size == 7:
        sigmaJ,C,P1,M1,K1,e1,omega1 = theta_real
        T01 = get_T0(P1, M1)
        h1, k1 = get_h_k(e1, omega1)
        model = model + get_rv2((P1,T01,K1,h1,k1), t)

    elif theta_real.size == 12:
        sigmaJ,C,P1,M1,K1,e1,omega1,P2,M2,K2,e2,omega2 = theta_real
        T01, T02 = get_T0(P1, M1), get_T0(P2, M2)
        h1, k1 = get_h_k(e1, omega1)
        h2, k2 = get_h_k(e2, omega2)
        model = model + get_rv2((P1,T01,K1,h1,k1), t)
        model = model + get_rv2((P2,T02,K2,h2,k2), t)

    elif theta_real.size == 17:
        sigmaJ,C,P1,M1,K1,e1,omega1,P2,M2,K2,e2,omega2,P3,M3,K3,e3,omega3 = theta_real
        T01, T02, T03 = get_T0(P1, M1), get_T0(P2, M2), get_T0(P3, M3)
        h1, k1 = get_h_k(e1, omega1)
        h2, k2 = get_h_k(e2, omega2)
        h3, k3 = get_h_k(e3, omega3)
        model = model + get_rv2((P1,T01,K1,h1,k1), t)
        model = model + get_rv2((P2,T02,K2,h2,k2), t)
        model = model + get_rv2((P3,T03,K3,h3,k3), t)

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
