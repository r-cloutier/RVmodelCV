'''
Write functions to compute useful quantities from RV and transit 
model parameters.
'''
from imports import *

# Define conversion functions
def days2sec(t):
    return t*24.*60*60
def sec2days(t):
    return t/(24.*60*60)
def days2yrs(t):
    return t/365.25
def yrs2days(t):
    return t*365.25
def Msun2kg(m):
    return m*1.98849925145e30
def kg2Msun(m):
    return m/1.98849925145e30
def Mearth2kg(m):
    return m*6.04589804468e24
def kg2Mearth(m):
    return m/6.04589804468e24
def Rearth2m(r):
    return r*6371e3
def m2Rearth(r):
    return r/6371e3
def Rsun2m(r):
    return r*695500e3
def m2Rsun(r):
    return r/695500e3
def AU2m(r):
    return r*1.495978707e11
def m2AU(r):
    return r/1.495978707e11
def pc2m(r):
    return r*3.08567758149137e16
def m2pc(r):
    return r/3.08567758149137e16 


def semimajoraxis(P, Ms, Mp):
    ''' Compute the semimajor axis in meters from Kepler's third law.

    P = orbital period in days
    Ms = stellar mass in Msun
    Mp = planet mass in Mearth'''
    P, Ms, Mp = days2sec(P), Msun2kg(Ms), Mearth2kg(Mp)
    G = 6.67e-11
    return (G*(Ms+Mp)*P*P/(4*np.pi*np.pi))**(1./3)

def period_sma(sma, Ms, Mp):
    '''Compute the orbital period in days, from the 
    semimajor axis.

    sma = semimajoraxis (AU)
    Ms = stellar amss in Msun
    Mp = planet mass in Mearth'''
    sma, Ms, Mp = AU2m(sma), Msun2kg(Ms), Mearth2kg(Mp)
    G = 6.67e-11
    return sec2days(np.sqrt(4*np.pi*np.pi*sma**3/(G*(Ms+Mp))))

def RV_K(P, Ms, Mp, ecc=0., inc=90.):
    '''Compute the RV semiamplitude in m/s

    P  = period in days
    Ms = stellar mass in Msun
    Mp = planet mass in MEarth
    ecc = eccentricity
    inc = inclination in degrees'''
    P, Ms, Mp, inc = days2sec(P), Msun2kg(Ms), Mearth2kg(Mp), np.deg2rad(inc)
    G = 6.67e-11
    return (2*np.pi*G/(P*Ms*Ms))**(1./3) * Mp*np.sin(inc) / unumpy.sqrt(1-ecc**2)

def RV_Mp(P, Ms, K, ecc=0., inc=90.):
    '''Compute the planet mass from RV semiamplitude in Earth masses

    P  = period in days
    Ms = stellar mass in Msun
    K = RV semiamplitude in m/s
    ecc = eccentricity
    inc = inclination in degrees'''
    P, Ms, inc = days2sec(P), Msun2kg(Ms), unumpy.radians(inc)
    G = 6.67e-11
    mp = K * (P*Ms*Ms/(2*np.pi*G))**(1./3) * unumpy.sqrt(1-ecc**2)/unumpy.sin(inc)
    return kg2Mearth(mp)

def impactparam_inc(a_Rs, inc, ecc=0., omega=0.):
    '''Compute the impact parameter from the inclination and scaled 
    semimajor axis.

    a_Rs = semimajor axis over stellar radius
    inc = inclination in degrees
    ecc = eccentricity
    omega = argument of periapsis in degrees'''
    inc, omega = np.deg2rad(inc), np.deg2rad(omega)
    return a_Rs * np.cos(inc) * ((1-ecc**2)/(1+ecc*np.sin(omega)))

def inclination(a_Rs, b, ecc=0., omega=0.):
    '''Compute the inclination from the impact parameter and scaled 
    semimajor axis. Inc is in radians.

    a_Rs = semimajor axis over stellar radius
    b = impact parameter
    ecc = eccentricity
    omega = argument of periapsis in degrees'''
    omega = unumpy.radians(omega)
    return unumpy.arccos(b / a_Rs * ((1+ecc*unumpy.sin(omega))/(1-ecc**2)))

def transit_width(P, a_Rs, D, b, ecc=0., omega=0.):
    '''Compute the transit width (duration) in the same units as the 
    input units of the orbital period P.

    a_Rs = semimajor axis over stellar radius
    D = transit depth; (Rp/Rs)**2
    b = impact parameter
    ecc = eccentricity
    omega = argument of periapsis in degrees'''
    return P/(np.pi*a_Rs) * np.sqrt((1+np.sqrt(D))**2 - b*b) * (np.sqrt(1-ecc**2)/(1+ecc*np.sin(omega)))

def impactparam_T(P, a_Rs, T, ecc=0., omega=0.):
    '''Compute the impact parameter from the transit width.

    P = orbital period in days
    a_Rs = semimajor axis over stellar radius
    T = transit width in days
    ecc = eccentricity
    omega = argument of periapsis in degrees'''
    return np.sqrt(1 - (np.pi*T/P * a_Rs * (1+ecc*np.sin(omega))/np.sqrt(1-ecc**2))**2)

def RM_K(Vsini, D):
    '''Compute the approximate semiamplitude for the Rossiter-McLaughlin 
    effect in m/s.

    Vsini = projected stellar rotation velocity in km/s
    D = transit depth'''
    return (Vsini*D / (1-D))*1e3

def logg_model(mp, rp):
    '''Compute the surface gravity from the planet mass and radius.

    mp = planet mass in Mearth
    rp = planet radius in Rearth'''
    mp, rp, G = Mearth2kg(mp), Rearth2m(rp), 6.67e-11
    return np.log10(G*mp/(rp*rp) * 1e2)

def logg_southworth(P, K, aRp, ecc=0., inc=90.):
    '''Compute the surface gravity in m/s^2 from the equation in Southworth 
    et al 2007.
    P = orbital period in days
    K = RV semiamplitude in m/s
    aRp = semimajor axis over planet radius
    ecc = eccentricity
    inc = inclination in degrees'''
    P, inc = days2sec(P), unumpy.radians(inc)
    return 2*np.pi*K*aRp*aRp * unumpy.sqrt(1-ecc*ecc) / (P*unumpy.sin(inc))
   
def tcirc(P, Mp, Ms, Rp):
    '''Compute the circularization timescale for a rocky planet 
    in years. From Goldreich & Soter 1966.

    P = orbital period in days
    Mp = planet mass in Earth masses
    Ms = stellar mass in Solar masses
    Rp = planet radius in Earth radii'''
    Q = 1e2   # for a rocky exoplanet
    tcirc = 2.*P*Q/(63*np.pi) * Mearth2kg(Mp)/Msun2kg(Ms) * (semimajoraxis(P,Ms,Mp) / Rearth2m(Rp))**5
    return tcirc / 365.25    # in years

def RVspot(Prot, Rs, photometricA=5e-3, dVc=4., kappa=10.):
    '''Estimate the radial velocity jitter from the flux effect 
    convective blueshift effect, and total of a rotating star 
    spot in m/s.
    Prot = rotation period in days
    Rs  = stellar radius in Rs'''
    dt = .01*Prot/25.
    t = np.arange(0, 2*Prot, dt)
    dmag = photometricA*np.sin(2*np.pi*t/Prot)
    flux = 10**(-.4*dmag)
    Psi0 = np.max(flux)
    Thetamin = np.min(flux)
    f = (Psi0 - Thetamin) / Psi0
    F = 1. - flux / Psi0
    Fdot = np.gradient(F, dt) / (24.*60*60)  # F/sec
    RVrot = -F*Fdot*Rsun2m(Rs)/f
    RVconv = F*F*dVc*kappa/f
    RVtot = RVrot + RVconv
    return np.max(abs(RVrot)), np.max(abs(RVconv)), np.max(abs(RVtot))  # m/s


def transit_spectroscopy_depth(Rs, mp, rp, Teff, mu):
    '''Compute the expected signal in transit spectroscopy in ppm given 
    the planet's mass (Mearth), radius (Rearth), Teff (K), mean molecular 
    weight and the stellar radius.'''
    kb, G, m = 1.38e-23, 6.67e-11, 1.67e-27
    g = G*Mearth2kg(mp) / (Rearth2m(rp)**2)
    depth = (Rearth2m(rp) / Rsun2m(Rs))**2
    H = kb*Teff / (mu*m*g)
    return 2e6*depth*H/Rearth2m(rp), H*1e-3

def rad2mass1(rad):
    '''Compute the mass of a planet from its radius in Earth radii assuming 
    a step-wise density of Earth-like for rp <= 1.6 Rearth and Neptune-like
    otherwise.'''
    rhoN_rhoE = 1.638 / 5.514
    rhoJ_rhoE = 1.326 / 5.514
    if rad <= 1.6:
	return rad**3
    elif rad > 1.6 and rad < 234:
	return rhoN_rhoE * rad**3
    else:
	return rhoJ_rhoE * rad**3

def rad2mass2(rp, maxE=1.5, maxN=4, mean_only=True):
    '''Convert the input radii in Earth radii to masses using the 
    analytical model from Weiss & Marcy 2014.'''
    rhoJ = 1.326  # cgs
    rmsE, rmsN = 2.7, 4.7      # scatter in Earth masses
    if rp <= maxE:
    	mp = .44*rp**3 + .614*rp**4
        if not mean_only:
            mptmp = mp + np.random.randn() * rmsE
            while mptmp < 0:
                mptmp = mp + np.random.randn() * rmsE
            mp = mptmp
    elif rp <= maxN:
        mp = 2.69 * rp**(.93)
        if not mean_only:
            mptmp = mp + np.random.randn() * rmsN
            while mptmp < 0:
                mptmp = mp + np.random.randn() * rmsN
            mp = mptmp
    else:
        mp = kg2Mearth(4*np.pi/3 * rhoJ * 1e-3 * (Rearth2m(rp)*1e2)**3)
    return mp



def radius2mass_max(rad):
    '''Compute the maximum planet mass given its radius at 99% confidence 
    from Weiss & Marcy 2014.'''
    if rad < 1.5:
	rho = 2.43 + 3.39*rad  # cgs
	rhoEarth = 5.51
	mass = rho/rho_Earth * rad**3
	sigma = 2.7
    elif 1.5 <= rad <= 4:
	mass =  2.69*rad**(.93)
	sigma = 4.7
    return mass + 3*sigma


def astrometric_K(P, Ms, Mp, dist):
    '''Compute the astrometric semiamplitude in microarcsec 
    P in days
    Ms in MSun
    Mp in Mearth
    dist in pc.'''
    P, Ms, Mp, dist = days2sec(P), Msun2kg(Ms), Mearth2kg(Mp), pc2m(dist)
    G = 6.67e-11
    Krad = (G*P*P/(4*np.pi*np.pi*Ms*Ms))**(1./3) * Mp /dist
    return np.rad2deg(Krad) * 60. * 60. * 1e6


def is_Lagrangestable(Ps, Ms, mps, eccs):
    '''Compute if a system is Lagrange stable (conclusion of barnes+
    greenberg 06).
    mp_i = Mearth'''
    Ps, mps, eccs = np.array(Ps), np.array(mps), np.array(eccs)
    smas = semimajoraxis(Ps, Ms, mps)
    stable = np.zeros(mps.size-1)
    for i in range(1, mps.size):
        mu1 = Mearth2kg(mps[i-1]) / Msun2kg(Ms)
        mu2 = Mearth2kg(mps[i]) / Msun2kg(Ms)
        alpha = mu1+mu2
        gamma1 = np.sqrt(1-float(eccs[i-1])**2)
        gamma2 = np.sqrt(1-float(eccs[i])**2)
        delta = np.sqrt(smas[i]/smas[i-1])
        deltas = np.linspace(1.000001, delta, 1e3)
        LHS = alpha**(-3.) * (mu1 + mu2/(deltas**2)) * \
              (mu1*gamma1 + mu2*gamma2*deltas)**2
        RHS = 1. + 3**(4./3) * mu1*mu2/(alpha**(4./3))
        fint = interp1d(LHS, deltas, bounds_error=False, fill_value=1e8)
        deltacrit = fint(RHS)
        stable[i-1] = True if delta >= 1.1*deltacrit else False
    return stable
