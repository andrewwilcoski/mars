"""
This module contains functions for calculating solar
angles from orbital elements

It has the additional feature of being able to handle insolation on sloped surfaces
"""

# Constants
AU = 1.49598261e11 # Astronomical Unit [m]
GM = 3.96423e-14 # G*Msun [AU**3/s**2]
TWOPI = 6.283185307

import numpy as np
import heat1d

#def orbitParams(a, ecc, obliq, omega, \
#                nu, dec, r, nudot):

def orbitParams(model):

    a = model.planet.rAU
    ecc = model.planet.eccentricity
    nu = model.nu
    obliq = model.planet.obliquity
    Lp = model.planet.Lp
    
    # Useful parameter:
    x = a*(1 - ecc**2)
    
    # Distance to Sun
    model.r = x/(1 + ecc*np.cos(nu))
    
    # Solar declination
    model.dec = np.arcsin( np.sin(obliq)*np.sin(nu+Lp) )
    
    # Angular velocity
    model.nudot = model.r**-2 * np.sqrt(GM*x)
    
def cosSolarZenith(lat, dec, h):
    
    # Cosine of solar zenith angle
    x = np.sin(lat)*np.sin(dec) + np.cos(lat)*np.cos(dec)*np.cos(h)
    
    # Clipping function = zero when sun below horizon:
    y = 0.5*(x + np.abs(x))
    
    return y

def hourAngle(t, P):
    
    return (TWOPI * t/P) % TWOPI

def cosSlopeSolarZenith(lat, dec, h, alpha, beta):
    
    # Cosine of solar zenith angle
    cosz = np.sin(lat)*np.sin(dec) + np.cos(lat)*np.cos(dec)*np.cos(h)
    sinz = np.sin(np.arccos(cosz))
    
    #Sine of solar elevation above a sloped surface with slope 'alpha' and azimuth of slope 'beta'
    #Positive 'alpha' with an azimuth of 0 degrees is a north-facing slope, and is south-facing when the azimuth is 180 degrees
    
    #First calculate solar azimuth
    arg = (np.sin(dec)-cosz*np.sin(lat)) / (sinz*np.cos(lat))
    #Make sure argument of np.arccos() is within [-1,1]
    if (arg < -1.):
        arg = -1.
    elif (arg > 1.):
        arg = 1.
        
    #Original calculation of solar azimuth angle. Worked for calculations, but was not correct.
    #if (h >= 0. and h <=np.pi):
    #    sa = np.arccos( arg )
    #if (h > np.pi and h <=2.*np.pi):
    #    sa = -np.arccos( arg )
        
    if (h >= 0. and h <=np.pi):
        sa = 2.*np.pi - np.arccos( arg )
    if (h > np.pi and h <=2.*np.pi):
        sa = np.arccos( arg )
    
    #Now calculate difference between solar azimuth and slope azimuth
    a = sa - beta
    
    #cosine of solar zenith above slope
    #if statement ensures that polar night will be observed
    if (cosz <= 0):
        cos_slope = 0
    else:
        cos_slope = np.cos(alpha)*cosz + np.sin(alpha)*sinz*np.cos(a)
    
    #Clipping function = zero when sun is below local slope horizon
    y = 0.5*(cos_slope + np.abs(cos_slope))
    
    return y, sa

def orbitParamsAtmo(model):

    a = model.planet.rAU
    ecc = model.planet.eccentricity
    nu = model.nu_atm
    obliq = model.planet.obliquity
    
    # Useful parameter:
    x = a*(1 - ecc**2)
    
    # Distance to Sun
    model.r_atm = x/(1 + ecc*np.cos(nu))
    
    # Angular velocity
    model.nudot_atm = model.r_atm**-2 * np.sqrt(GM*x)