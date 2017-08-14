'''Ryan Cloutier

Compute the min(period) and max(period) of a planet in the habitable zone 
around a planet with an intrinsic luminosity L and mass M. Assuming 
circular planet orbit and zero planetary albedo.

Last updated: June 29, 2015
'''
import numpy as np
from scipy.interpolate import LinearNDInterpolator as lint
import rvs 

# Define conversion functions
def Msun2kg(m):
    return m*1.989e30
def Lsun2W(l):
    return l*3.846e26

# Get Ls from Ms
def get_baraffe_L_R_T(Ms):
    baraffe=np.loadtxt('/Users/ryancloutier/anaconda2/lib/python2.7/site-packages/BCAH98.dat')
    barmass=baraffe[:,0]    # Msun
    barage=baraffe[:,1]     # Gyrs
    barTeff=baraffe[:,2]    # K
    barL=baraffe[:,4]       # log(L/Lsun)   IS THIS LOG10 OR LN??
    # Interpolate to get Teff
    lint_Teff=lint(np.array([barmass,barage]).T,barTeff)
    age=2.   # Gyrs
    Teff=lint_Teff(Ms,age)
    # Interpolate to get L
    lint_L=lint(np.array([barmass,barage]).T,barL)
    L=10**(lint_L(Ms,age))   # Ls
    sigma=5.67037e-8
    Rout=np.sqrt(L*3.846e26/(4*np.pi*sigma*Teff*Teff*Teff*Teff))  # meters
    return L,Rout,Teff


# Get min/max orbital distances (m)
def get_distances(L):
    Teff=np.array([373.,273.])
    sigma=5.670e-8
    L=Lsun2W(L)
    return np.sqrt(L/(16*np.pi*sigma*Teff**4))

# Get min/max orbital periods (days)
def get_periods(M):
    L,R=get_baraffe_L_R_T(M)
    G=6.67e-11
    M=Msun2kg(M)
    d=get_distances(L)
    return tuple(np.sqrt(4*np.pi*np.pi/(G*M)*d*d*d)/(24*60*60))

def get_Kopparapu_HZPlims(Ms, Teff=0):
    '''Use the equations from Kopparapu et al 2013 to compute the inner 
    (moist GH) and outer (max GH) edges of the HZ in days.'''
    # Get Teff if not given
    if Teff == 0:
	Teff = float(get_baraffe_L_R_T(Ms)[2])
    # Define coefficients for the (inner, outer) edges from Kopparapu13
    SeffSun = np.array((1.014, .3438))
    a = np.array((8.1774e-5, 5.8942e-5))
    b = np.array((1.7063e-9, 1.6558e-9))
    c = np.array((-4.3241e-12, -3.0045e-12))
    d = np.array((-6.6462e-16, -5.2983e-16))
    Ts = Teff - 5780.
    # Compute the inner and outer periods
    Seff = SeffSun + a*Ts + b*Ts**2 + c*Ts**3 + d*Ts**4
    Ls, Rs, Teffbaraffe = get_baraffe_L_R_T(Ms)  # Lsun
    dist = np.sqrt(Ls/Seff)
    return rvs.period_sma(dist, Ms, 0.)


if __name__ == '__main__':
    Ms=.12
    #Ls=.0009
    pmin,pmax=get_periods(Ms)
    print '\nMinimum and maximum periods within the HZ (days):'
    print '%.2f' %pmin
    print '%.2f' %pmax
