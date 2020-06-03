"""
1-d thermal modeling functions

With ability to handle surface fluxes on sloped surfaces and under atmospheric conditions

This is the cleaned up version of 'heat1d_surface_atmo4.py'.

I've tried to cut out uneccessary code and add more comments for clarity.

Created 05/05/2020
"""

# NumPy is needed for various math operations
import numpy as np

# Physical constants:
sigma = 5.67051196e-8 # Stefan-Boltzmann Constant
kb = 1.38066e-23 #Boltzmann's Constant [J.K-1]
#S0 = 1361.0 # Solar constant at 1 AU [W.m-2]
chi = 2.7 # Radiative conductivity parameter [Mitchell and de Pater, 1994]
R350 = chi/350**3 # Useful form of the radiative conductivity
Rg = 8.314 #universal gas constant [J.K-1.mol-1]
GM = 3.96423e-14 # G*Msun [AU**3/s**2]

#CO2 ice parameters
L_CO2 = 6e5 #latent heat of CO2 [J.kg-1] (Aharonson and Schorghofer 2006)
A_CO2 = 0.6 #0.40 #0.65 #Albedo of CO2 frost (Aharonson and Schorghofer 2006)
emis_CO2 = 0.9 #emissivity of CO2 frost (Bapst 2018)
mc = 0.044 #molar mass of CO2 [kg]

#H2O ice parameters
L_H2O = 2.83e6 #latent heat of H2O [J.kg-1] (Dundas and Byrne 2010)
M_w = 2.99e-26 #molecular mass of water [kg]
A_drag = 0.002 #drag coefficient (Dundas and Byrne 2010)
Hc = 1e4 #estimate of Mars northern summer water vapor condensation level [m] from (Smith 2002)
h_vap = 5e-5 #estimate of Mars northern summer water vapor column abundance [m] from (Smith 2002)
rho_liq = 997 #density of liquid water [kg.m-3]`
mw = 0.018 #molar mass of water [kg]
ki = 3.2 #conductivity of ice [W.m-1.K-1]
rho_ice = 920. #density of water ice [kg.m-3] at around 200K
cp0_ice = 1540. #heat capacity of ice

#Mars atmospheric properties
#rho_surf = 0.020 #mean Martian atmospheric surface density [kg.m-3]
H_atm = 7. #11.1 #scale height of Mars atmosphere (at poles) [km]
k_atm = 1e-2 #Thermal Conductivity of Martian atmosphere [W.m-1.K-1] (estimate)
cp_atm = 736 #specific heat of CO2 at constant pressure [J.kg-1.K-1] (Savijarvi 1995)
d_lam = 0.01 #thickness of laminar layer in atmosphere [m]

#Wind profile constants
k_wind = 0.4 #Von Karman constant
z0 = 0.001 #Surface roughness of ice [m]
z_vik = 1.6 #Height at which Viking lander wind measurements were taken [m]

#Atmospheric model grid parameters
h_max = 6e3 #Maximum height of atmospheric spatial grid [m]
#N_atm = 4 #Number of points in atmospheric spatial grid
Omega = 2.*np.pi/(24.6229*60*60) #Mars rotation rate [rad.s-1]

# Numerical parameters:
F = 0.5 # Fourier Mesh Number, must be <= 0.5 for stability
m = 10 # Number of layers in upper skin depth [default: 10]
n = 5 # Layer increase with depth: dz[i] = dz[i-1]*(1+1/n) [default: 5]
b = 20 # Number of skin depths to bottom layer [default: 20]

# Accuracy of temperature calculations
# The model will run until the change in temperature of the bottom layer
# is less than DTBOT over one diurnal cycle
DTSURF = 0.1 # surface temperature accuracy [K]
DTBOT = DTSURF # bottom layer temperature accuracy [K]
NYEARSEQ = 1 # equilibration time [orbits]
NPERDAY = 24 # minimum number of time steps per diurnal cycle
Ddmdt = 1e-7 #CO2 frost flux accuracy [kg.m-2.s-1]

# Pandas is needed to read the Mars optical depth file
# interp2d is used to interpolate optical depth based on latitude and Ls
import pandas as pd
from scipy.interpolate import interp2d, interp1d
from scipy import integrate

# MatPlotLib and Pyplot are used for plotting
import matplotlib.pyplot as plt
import matplotlib as mpl

# Methods for calculating solar angles from orbits
import orbits_slopes

# Useful thermodynamics for planets
import planetThermo

# Planets database
import planets
g_mars = planets.Mars.g

from numba import jitclass, jit
from numba import int32, float32, boolean, deferred_type

import os
import shutil

import warnings
import time

import multiprocessing as mp

# Models contain the profiles and model results
#@jitclass(spec)
class model(object):
    
    # Initialization
    def __init__(self, planet=planets.Mars, lat=0, ndays=1, Ls=0., x=np.array([0,0]), alpha=0, beta=0, f_IR=0.04, f_scat=0.02, \
                 u_wind=3., b_exp=0.2, atmExt=True, elev=-4.23, t_obliq=0, dt=0, A_ice=0.32, \
                 dx=0.1, shadow=True, yaxis_ext=True, por=0.5, z_rho=10., density='linear', N_atm=8, N_skin=4, Ls_eq=0.,\
                 vap_set=1, vap_set1=1, vap_Ls_start=0, vap_Ls_end=0):
        
        self.landcount = 0
        
        self.T_land_min = 200. #diurnal minimum of horizontal surface temperature
        
        # Initialize
        self.planet = planet
        self.lat = lat
        self.Sabs = self.planet.S
        self.r = self.planet.rAU # solar distance [AU]
        
        #t_obliq in thousands of Earth-years from present (e.g. t_obliq=-1 == 1000 years before present)
        self.planet.eccentricity, self.planet.obliquity, self.planet.Lp = MarsOrbitalParams(t_obliq)
        
        print ('ecc:', self.planet.eccentricity)
        print ('obliq:', self.planet.obliquity*180./np.pi)
        print ('Lp:', self.planet.Lp*180./np.pi)
        
        self.Ls = Ls #Ls at which to start simulation [deg]
        self.Ls_eq = Ls #####Ls_eq #Ls at which to start equilibration [deg]
        self.nu = self.Ls_eq*(np.pi/180.) - self.planet.Lp # orbital true anomaly [rad]
        self.nuout = self.Ls*(np.pi/180.) - self.planet.Lp # orbital true anomaly [rad] at which simulation starts
        self.nudot = np.float() # rate of change of true anomaly [rad/s]
        
        #Additional orbital parameters for atmospheric timestep
        self.r_atm = self.planet.rAU # solar distance [AU]
        self.nu_atm = self.Ls*(np.pi/180.) - self.planet.Lp # orbital true anomaly [rad]
        self.nudot_atm = np.float() # rate of change of true anomaly [rad/s]
        
        self.dec = np.float() # solar declination [rad]
        self.beta = beta #azimuth of slope surface [rad]
        self.h_noon = np.float() #hour angle in previous simulation run
        self.c_noon = np.float() #cosine of solar zenith angle at noon of most recent day in simulation
        self.shadow = shadow
        self.theta = np.float()
        self.f_IR = f_IR #fraction of noontime insolation that contributes to atmospheric IR surface flux
        self.f_scat = f_scat # fraction of scattered light from atmosphere
        self.u_wind = u_wind #wind speed for forced water ice sublimation [m.s-1]
        self.T_atm = 200. #Temperature of atmosphere that will affect H2O sublimation
        self.b = b_exp #parameter that affects dependence of T_atm on T_land_min and T_land
        self.h = np.float() #hour angle
        self.tau = np.float()
        self.atmExt = atmExt # whether or not to include atmospheric extinction. default False
        self.elev = elev
        self.A_ice = A_ice #Albedo of ice
        self.T_CO2_frost = np.float()
        self.N_atm = N_atm
        self.N_skin = N_skin
        
        #Setting vapor density (Optional, for exploring atmospheric vapor conditions)
        self.vap_set = vap_set #water vapor density
        self.vap_set1 = vap_set1 #water vapor density
        self.vap_Ls_start = vap_Ls_start #Ls in degrees
        self.vap_Ls_end = vap_Ls_end #Ls in degrees
        
        #ice density vs depth parameters
        self.por = por #porosity of ice at surface
        self.z_rho = z_rho #depth below which ice density is that of slab ice (920 kg.m-3) (e-folding scale when density=exp)
        self.density = density
        
        # Need indicies to differentiate points on surface that have CO2 frost from those that do not
        self.idx_clear = np.float()
        self.idx_frost = np.float()
        self.idx_shadow = np.float()
        
        # Point positions and slopes
        self.x = x #position of points in surface [x,y] in m
        self.alpha = alpha #slope of surface [rad] (positive slopes are pole-facing)
        self.los_idx = [] #line of sight index for each point of the surface
        self.N_x = self.x.shape[1]
        self.dx = dx
        
        # Useful factors for density ratio calculation
        self.x_dp = np.zeros(self.N_x)
        self.y_dp = np.zeros(self.N_x)
        
        # Calculate view factor for each point on surface
        self.yaxis_ext = yaxis_ext
        self.F_los, self.los_idx, self.idx_max, self.idx_min = viewFactors(self)
        
        # Initialize albedo array
        self.f_alb = np.ones(self.N_x)*(1 - self.A_ice)
        self.Albedo = np.ones(self.N_x)*self.A_ice
        
        # Initialize emissivity array
        self.emissivity = self.planet.emissivity * np.ones(self.N_x)
        
        # Initialize Mars Optical Depth and Atmospheric Temperature Functions
        self.OpDepth = MarsOpticalDepth()
        self.AtmTemp = MarsAtmTemps()
        
        # Initialize arrays
        self.Qs = np.zeros(self.N_x) # surface flux
        self.Q_solar = np.zeros(self.N_x) # solar flux
        self.Q_IR = np.zeros(self.N_x) # atmospheric IR emission
        self.Q_scat = np.zeros(self.N_x) # atmospheric scattering
        self.Q_surf = np.zeros(self.N_x) # surface IR and visible flux
        self.Q_surf_vis = np.zeros(self.N_x) # reflected visible component of Q_surf
        self.Q_surf_IR = np.zeros(self.N_x) # reradiated IR component of Q_surf
        self.c = np.zeros(self.N_x) #cosSolarZenith
        self.cosz = np.float()
        self.sa = np.float() #Solar Azimuth Angle
        
        # Initialize model profile
        self.profile = profile(planet, lat, u_wind, self.N_x, self.emissivity, self.por, \
                               self.z_rho, self.density, self.elev, self.Ls, self.N_atm, self.N_skin)
        
        # Model run times
        # Equilibration time -- TODO: change to convergence check
        self.equiltime = NYEARSEQ*planet.year - \
                        (NYEARSEQ*planet.year)%planet.day
        self.equildone = False
        # Run time for output
        self.endtime = self.equiltime + ndays*planet.day
        self.t = 0.
        if (dt == 0):
            self.dt = getTimeStep(self.profile, self.planet.day)
        elif (dt == 1):
            self.dt = 2.*getTimeStep(self.profile, self.planet.day)
        else:
            self.dt = dt
        
        #Atmospheric model timestep for multistepping model
        self.dt_atm = getTimeStepAtmo(self.profile, self.dt)
        
        print ('Pre-Equilibration dt:', self.dt)
        print ('Pre-Equilibration dt_atm:', self.dt_atm)
        # Check for maximum time step
        self.dtout = self.dt
        dtmax = self.planet.day/NPERDAY
        if self.dt > dtmax:
            self.dtout = dtmax
        
        # Array for output temperatures and local times
        self.N_steps = np.int( (ndays*planet.day)/self.dtout )
        self.N_day = np.int( (planet.day)/self.dtout ) #number of steps in a day
        self.N_z = np.size(self.profile.z, axis=0)
        self.T = np.zeros([self.N_steps, self.N_z, self.N_x])
        self.Qst = np.zeros([self.N_steps, self.N_x]) #Qs as a funciton of t
        self.Q_solart = np.zeros([self.N_steps, self.N_x])
        self.Q_IRt = np.zeros([self.N_steps, self.N_x])
        self.Q_scatt = np.zeros([self.N_steps, self.N_x])
        self.Q_surft = np.zeros([self.N_steps, self.N_x])
        self.m_CO2t = np.zeros([self.N_steps, self.N_x])
        self.m_H2Ot = np.zeros([self.N_steps, self.N_x])
        self.m_H2O_free = np.zeros([self.N_steps, self.N_x])
        self.m_sub = np.zeros([self.N_steps, self.N_x])
        self.m_mol = np.zeros([self.N_steps, self.N_x])
        self.rho_atm = np.zeros([self.N_steps, self.N_atm])
        self.rho_vap = np.zeros([self.N_steps, self.N_atm])
        self.rho_sat = np.zeros([self.N_steps, self.N_x])
        self.nut = np.zeros([self.N_steps])
        self.Lst = np.zeros([self.N_steps])
        
    def run(self):
        
        start = time.time()
        # Equilibrate the model
        while (self.t < self.equiltime):
            self.advance()
            if (self.landcount == self.T.shape[0]):
                self.landcount = 0 #reset T_land index if it exceeds size of T_land
            self.T[self.landcount,:,:] = self.profile.T # Need some record of past temperatures for T_land when alpha=0
        self.equildone = True
        end = time.time()
        print ('Equilibration Time:', end-start)
        # Run through end of model and store output
        self.dt = self.dtout
        #Atmospheric model timestep
        self.dt_atm = getTimeStepAtmo(self.profile, self.dt)
        print ('dt:', self.dt)
        print ('dt_atm:', self.dt_atm)
        self.t = 0. # reset simulation time
        self.landcount = 0 #reset T_land index so T_land simulation times align with slope simulation times
        self.T = np.zeros([self.N_steps, self.N_z, self.N_x]) # Reset temperature array to zero
        self.profile.m_H2O = np.zeros(self.N_x) #reset mass of H2O ice lost
        self.nu = self.nuout
        self.nu_atm = self.nuout
        
        print ('START SIM: ', self.nu)
        for i in range(0,self.N_steps):
            self.advance()
            self.T[i,:,:] = self.profile.T # temperature [K]
            self.Qst[i, :] = self.Qs # Total surface flux
            self.Q_solart[i, :] = self.Q_solar
            self.Q_IRt[i, :] = self.Q_IR
            self.Q_scatt[i, :] = self.Q_scat
            self.Q_surft[i, :] = self.Q_surf
            self.m_CO2t[i, :] = self.profile.m_CO2
            self.m_H2Ot[i, :] = self.profile.m_H2O
            self.m_H2O_free[i, :] = self.profile.m_free
            self.m_sub[i, :] = self.profile.m_sub
            self.m_mol[i, :] = self.profile.m_mol
            self.rho_atm[i, :] = self.profile.rho_atm
            self.rho_vap[i, :] = self.profile.rho_vap
            self.rho_sat[i, :] = self.profile.rho_sat
            self.nut[i] = self.nu
            self.Lst[i] = self.Ls
        
        print ('END SIM: ', self.nu)
            
    def advance(self):
        self.updateOrbit()
        
        #Calculate surface flux
        self.tau = self.OpDepth(self.Ls, self.lat*180./np.pi)
        T_IR_atm = self.AtmTemp(self.Ls, self.lat*180./np.pi)
        
        self.emissivity[:] = self.planet.emissivity
        self.emissivity[np.where(self.profile.m_CO2 > 0)] = emis_CO2
        
        self.Qs, self.Q_solar, self.Q_IR, self.Q_scat, self.Q_surf, self.h_noon, self.c_noon = self.surfFlux(self.t,\
                 self.planet.day, self.lat, self.dec,\
                 self.alpha, self.beta, self.h_noon, self.c_noon, self.f_alb, self.A_ice, self.Albedo, self.profile.m_CO2,\
                 self.Ls, self.tau, self.atmExt, self.Q_solar, self.Sabs, self.r, self.planet.rAU,\
                 self.Q_scat, T_IR_atm, self.Q_IR, self.f_IR, self.shadow, self.idx_max, self.idx_min, self.Q_surf,\
                 self.Q_surf_vis, self.Q_surf_IR, self.F_los, self.los_idx, self.profile.T[0,:], self.emissivity, self.N_x,\
                 self.Qs, self.x)
        
        #Determine CO2 Frost Point Temperature based on surface atmospheric pressure
        self.profile.P_atm = P_atm(self.Ls, self.profile.elev+(self.profile.h*1e-3)) #atmospheric pressure profile
        self.T_CO2_frost = planetThermo.tco2(self.profile.P_atm[0]*0.95)
            
        ################### THIS VERSION IS CORRECT ###################
        #Determine which points on surface are covered in CO2 frost
        self.idx_frost = np.where( ( (self.profile.T[0, :] <= self.T_CO2_frost) & \
                                    np.logical_not((self.profile.m_CO2 <= 0) & (self.profile.dmdt < 0)) ) | \
                                  (self.profile.m_CO2 > 0) )
        self.profile.T[0, self.idx_frost] = self.T_CO2_frost
        self.idx_clear = np.where( (self.profile.T[0, :] > self.T_CO2_frost) | \
                                  ((self.profile.m_CO2 <= 0) & (self.profile.dmdt < 0)) )
        ###############################################################

        self.profile.m_CO2[self.idx_clear] = 0
        self.profile.dmdt[self.idx_clear] = 0

        self.idx_clear = self.idx_clear[0]
        self.idx_frost = self.idx_frost[0]

        #Frost-free water ice temperature
        self.profile.T[:,self.idx_clear], self.profile.T_bl[self.idx_clear] = \
                                self.profile.update_T_ice(self.dt, self.Qs[self.idx_clear], self.planet.Qb,\
                                self.profile.T[:,self.idx_clear], self.profile.T_bl[self.idx_clear], self.T_atm,\
                                self.profile.P_atm[0], self.profile.g1, self.profile.g2, self.profile.k[:,self.idx_clear],\
                                self.profile.cp[:,self.idx_clear], self.profile.rho[:,self.idx_clear],\
                                self.profile.rho_vap[0], self.profile.rho_atm[0], self.profile.rho_sat[self.idx_clear],\
                                self.profile.emissivity[self.idx_clear], self.profile.kc[0,self.idx_clear],\
                                self.profile.dz, self.profile.m_sub[self.idx_clear], self.profile.SH[self.idx_clear],\
                                self.profile.e_sat[self.idx_clear], self.x_dp[self.idx_clear], self.y_dp[self.idx_clear])

        #CO2 Frosted water ice flux
        self.profile.T[:,self.idx_frost], self.profile.T_bl[self.idx_frost], self.profile.m_CO2[self.idx_frost],\
        self.profile.dmdt[self.idx_frost] = self.profile.CO2Flux_ice(self.dt,\
                                self.Qs[self.idx_frost], self.planet.Qb, self.profile.T[:,self.idx_frost],\
                                self.profile.T_bl[self.idx_frost], self.T_atm, self.profile.P_atm[0],\
                                self.profile.g1, self.profile.g2, self.profile.k[:,self.idx_frost],\
                                self.profile.cp[:,self.idx_frost], self.profile.rho[:,self.idx_frost],\
                                self.profile.rho_vap[0], self.profile.rho_atm[0], self.profile.rho_sat[self.idx_frost],\
                                self.profile.emissivity[self.idx_frost], self.profile.kc[0,self.idx_frost],\
                                self.profile.dz, self.profile.m_sub[self.idx_frost], self.profile.SH[self.idx_frost],\
                                self.profile.e_sat[self.idx_frost], self.profile.m_CO2[self.idx_frost],\
                                self.profile.dmdt[self.idx_frost], self.x_dp[self.idx_frost], self.y_dp[self.idx_frost])

        #Set temperature of lowest atmospheric layer as mean of all surface temperatures across surface
        self.T_atm = np.mean(self.profile.T[0, :])
        #Set atmospheric temperature profile
        self.profile.T_atm = self.T_atm - 4.5e-3*self.profile.h #atmospheric temperature profile
        self.profile.T_atm[np.where(self.profile.T_atm < 0)] = 0

        self.profile.e_sat = satVaporPressureIce(self.profile.T[0, :])
        self.profile.rho_sat = (self.profile.e_sat*mw)/(Rg*self.profile.T[0, :])
        
        
        self.profile.update_cp()
        self.profile.update_k()
        
        #Update atmospheric model according to the atmospheric time step
        #Attempting to do this the quick way (jit), so need to pass this function alllllll the variables/arrays it needs.
        #Need to create cbrt array to hold result of numpy ufunc. Array creation not supported in numba nopython mode.
        cbrt = np.zeros_like(self.profile.rho_atm[0:-1]) 
        self.profile.e_vap = self.profile.WaterVolMix(self.Ls, self.lat*180./np.pi) * self.profile.P_atm[-1] * \
                             self.profile.fH2O_obliq
        if (self.profile.e_vap < 0):
            self.profile.e_vap = np.array([0])
        
        visc = kineViscosity(self.profile.T_bl, self.profile.P_atm[0])
        P = self.profile.P_atm[0]
        Ta = self.profile.T_atm[0]
        Ts = self.profile.T[0, :]
        e_sat = self.profile.e_sat
        self.x_dp = mc*P*(1./Ta - 1./Ts) + (mc - mw)*e_sat/Ts
        self.y_dp = 0.5*mc*P*(1./Ta + 1./Ts) - 0.5*(mc - mw)*e_sat/Ts
        
        drho = np.zeros_like(self.profile.rho_sat)
        rho_surf = ((self.profile.P_atm[0]*mc)/(Rg*self.profile.T[0,:])) - ((mc/mw)-1)*self.profile.rho_sat
        #rho_surf = ( ( ((self.profile.P_atm[0]*mc)/(Rg*self.profile.T[0,:])) - ((mc/mw)-1)*self.profile.rho_sat ) + \
        #           ( ((self.profile.P_atm[0]*mc)/(Rg*self.profile.T[0,:])) - ((mc/mw)-1)*self.profile.rho_vap[0] ) )/2.
        
        D = diffusionCoeff(self.profile.T_bl, self.profile.P_atm[0])
        D_atm = diffusionCoeff(self.profile.T_atm[:-1], self.profile.P_atm[:-1])
        visc_atm = kineViscosity(self.profile.T_atm[:-1], self.profile.P_atm[:-1])
        
        f_dp = (g_mars/(visc**2)) * (visc/D)
        
        (self.dt, self.dt_atm, self.profile.m_H2O, self.profile.m_sub, self.profile.m_mol, self.profile.m_free,\
         self.profile.T_atm, self.profile.P_atm, self.profile.d, self.profile.dh,\
         self.profile.rho_vap, self.profile.rho_atm, self.profile.rho_sat, self.profile.Kappa, self.profile.a, self.profile.b,\
         self.planet.rAU, self.planet.eccentricity, self.nu_atm, self.planet.Lp,\
         self.planet.obliquity, self.r_atm, self.nudot_atm, self.profile.e_vap, cbrt, drho) = self.updateAtmo(self.dt, self.dt_atm,\
                        self.profile.m_H2O, self.profile.m_sub, self.profile.m_mol, self.profile.m_free,\
                        self.profile.T_atm, self.profile.P_atm, self.profile.d, self.profile.dh,\
                        self.profile.rho_vap, self.profile.rho_atm, self.profile.rho_sat, self.profile.Kappa, self.profile.a,\
                        self.profile.b, self.planet.rAU, self.planet.eccentricity,\
                        self.nu_atm, self.planet.Lp, self.planet.obliquity, self.r_atm, self.nudot_atm, self.profile.e_vap,\
                        cbrt, visc, D, D_atm, visc_atm, rho_surf, drho, f_dp, self.vap_set, self.vap_set1, self.vap_Ls_start,\
                        self.vap_Ls_end)
        
        self.t += self.dt # Increment time
    
    def updateOrbit(self):
        orbits_slopes.orbitParams(self)
        self.nu += self.nudot * self.dt
        self.r_atm = self.r
        self.nu_atm = self.nu
        self.nudot_atm = self.nudot
        self.Ls = (self.nu + self.planet.Lp)*180./np.pi
        if (self.Ls >= 360.):
            self.Ls = self.Ls - 360.
    
    # Surface heating rate
    # May include solar and infrared contributions, reflectance phase function
    @staticmethod
    def surfFlux(t, day, lat, dec, alpha, beta, h_noon, c_noon, f_alb, A_ice, Albedo, m_CO2, Ls, tau, atmExt, Q_solar,\
                 Sabs, r, rAU, Q_scat, T_IR_atm, Q_IR, f_IR, shadow, idx_max, idx_min, Q_surf, Q_surf_vis,\
                 Q_surf_IR, F_los, los_idx, T, emissivity, N_x, Qs, x):
        h = hourAngle(t, day) # hour angle
        #cosine of incidence angle and solar azimuth angle
        c, sa = cosSlopeSolarZenith(lat, dec, h, alpha, beta)
        cosz = cosSolarZenith(lat, dec, h) #cosine of incidence angle for horizontal surface
        
        if (h < h_noon):
            #noon cosSolarZenith reset every day for use in atmospheric IR flux term
            c_noon = cosz
        h_noon = h
        
        i = np.arccos(c) # solar incidence angle [rad]
        
        #Albedo
        Albedo[:] = A_ice
        Albedo[np.where(m_CO2 > 0)] = A_CO2
        f_alb[:] = (1.0 - Albedo[:])
        
        #Insolation and Visible Atmospheric Scattering
        #0.04 term comes from maximum path length being limited by radius of curvature of planet (Aharonson and Schorghofer 2006)
        if (atmExt == True and cosz > 0.04):
            #Insolation
            Q_solar = f_alb * Sabs * (r/rAU)**-2 * c * np.exp(-tau/cosz)
            
            #Visible Atmospheric Scattering
            f_scat = 1. - np.exp(-0.9*tau/cosz)
            Q_scat = 0.5 * f_scat * f_alb * Sabs * (r/rAU)**-2 * cosz * \
                            np.cos(alpha/2.)**2
            
        elif (atmExt == True and cosz <= 0.04):
            #Insolation
            Q_solar = f_alb * Sabs * (r/rAU)**-2 * c * np.exp(-tau/0.04)
            
            #Visible Atmospheric Scattering
            f_scat = 1 - np.exp(-0.9*tau/0.04)
            Q_scat = 0.5 * f_scat * f_alb * Sabs * (r/rAU)**-2 * cosz * \
                            np.cos(alpha/2.)**2
        else:
            Q_solar = f_alb * Sabs * (r/rAU)**-2 * c
            
            if (cosz > 0):
                Q_scat = 0.5 * f_scat * Sabs * \
                         (r/rAU)**-2 * np.cos(alpha/2.)**2
            else:
                Q_scat[:] = 0.
        
        #Atmospheric IR emission
        tau_IR = tau*0.5 #0.14 #visible/IR optical depth factor of 0.5 from Forget 1998.
        eps = 1 - np.exp(-tau_IR)
        if (atmExt == True):
            Q_IR = f_IR * Sabs * (r/rAU)**-2 * np.cos(alpha/2.)**2 * c_noon + eps*sigma*T_IR_atm**4
        else:
            Q_IR = f_IR * Sabs * (r/rAU)**-2 * np.cos(alpha/2.)**2 * c_noon
        
        #Shadowing. Determine which points are shadowed by surrounding terrain.
        if (shadow == True):
            idx_shadow, theta = Shadowfax(x, sa, idx_max, idx_min, cosz)
            Q_solar[idx_shadow] = 0
        
        
        #Reflected and Emitted Flux from Surface within Line of Sight
        Q_surf, Q_surf_vis, Q_surf_IR = groundFlux(F_los, los_idx, T, emissivity, Q_solar, Albedo, N_x, Q_surf_vis, Q_surf_IR,\
                                                   Q_surf)
        
        #Total Incident Flux
        Qs = Q_solar + Q_IR + Q_scat + Q_surf
        
        return Qs, Q_solar, Q_IR, Q_scat, Q_surf, h_noon, c_noon
    
    @staticmethod
    @jit(nopython=True)
    def updateAtmo(dt, dt_atm, m_H2O, m_sub, m_mol, m_free, T_atm, P_atm, d, dh, rho_vap, rho_atm, rho_sat, Kappa, a, b, rAU,\
                   ecc, nu_atm, Lp, obliq, r_atm, nudot_atm, e_vap, cbrt, visc, D, D_atm, visc_atm, rho_surf, drho, f_dp,\
                   vap_set, vap_set1, vap_Ls_start, vap_Ls_end):
        
        n = int(dt/dt_atm)
        for i in range(0, n):
            
            #Calculation of Ls
            Ls = (nu_atm + Lp)*180./np.pi
            if (Ls >= 360.):
                Ls = Ls - 360.
                
            #(Optional, set upper water vapor density based on Ls)
            if ((Ls >= vap_Ls_start) and (Ls <= vap_Ls_end)):
                vap_set = vap_set1
                
            #Update water vapor density profile
            dt_atm, m_H2O, m_sub, m_mol, m_free, T_atm, P_atm, d, dh, rho_vap, rho_atm, rho_sat, e_vap, Kappa, a, b,\
            cbrt, drho = vapor_density_profile(dt_atm, m_H2O, m_sub, m_mol, m_free, T_atm, P_atm, d, dh, rho_vap, rho_atm,\
                                               rho_sat, e_vap, Kappa, a, b, cbrt, visc, D, D_atm, visc_atm, rho_surf, drho,\
                                               f_dp, vap_set)
            
            #update orbit for each atmospheric timestep
            nu_atm, nudot_atm = updateOrbitAtmo(dt_atm, rAU, ecc, nu_atm, obliq, r_atm, nudot_atm)
            
        return (dt, dt_atm, m_H2O, m_sub, m_mol, m_free, T_atm, P_atm, d, dh, rho_vap, rho_atm, rho_sat, Kappa, a,\
                b, rAU, ecc, nu_atm, Lp, obliq, r_atm, nudot_atm, e_vap, cbrt, drho)     

class profile(object):
    """
    Profiles are objects that contain the model layers
    
    The profile class defines methods for initializing and updating fields
    contained in the model layers, such as temperature and conductivity.
    
    """
    
    def __init__(self, planet=planets.Moon, lat=0, u_wind=0, N_x=1, emissivity=0.95, \
                 por=0.5, z_rho=10., density='linear', elev=-4.23, Ls=0., N_atm=8, N_skin=4):
        
        self.planet = planet
        self.lat = lat
        self.u_wind = u_wind
        self.T_bl = np.zeros(N_x) #boundary layer temperature
        self.T_atm = np.zeros(N_x)
        self.Ls = Ls
        self.elev = elev
        self.Patm = np.float()
        self.WaterVolMix = MarsWaterVolumeMixingRatio()
        self.e_sat = np.zeros(N_x)
        self.e_vap = np.float()
        self.SH = np.zeros(N_x) #sensible heat
        self.N_x = N_x
        self.idx_clear = np.float()
        self.idx_frost = np.float()
        self.N_atm = N_atm
        self.N_skin = N_skin
        self.dp = np.float()
        self.rho_sat = np.zeros(N_x)
        
        # Density Profile
        self.por = por
        self.z_rho = z_rho
        self.density = density
        
        # Initialize average water vapor pressure for current year
        #Present annual average H2O vapor partial pressure given by partialPressureObliq() for normalization
        PH2O_present = partialPressureObliq(25.19)
        #Fraction of present day H2O partial pressure at given obliquity
        self.fH2O_obliq = partialPressureObliq(planet.obliquity*180./np.pi)/PH2O_present
        
        # The spatial grid
        self.emissivity = emissivity
        ks = planet.ks
        kd = planet.kd
        rhos = planet.rhos
        rhod = planet.rhod
        H = planet.H
        
        cp0 = cp0_ice
        kappa = ki/(rho_ice*cp0)
        
        self.z = spatialGrid(skinDepth(planet.day, kappa), m, n, b, self.N_x)
        self.nlayers = np.shape(self.z)[0] # number of model layers
        self.dz = np.diff(self.z[:,0])
        self.d3z = self.dz[1:]*self.dz[0:-1]*(self.dz[1:] + self.dz[0:-1])
        self.g1 = 2*self.dz[1:]/self.d3z[0:] # A.K.A. "p" in the Appendix
        self.g2 = 2*self.dz[0:-1]/self.d3z[0:] # A.K.A. "q" in the Appendix
        
        # Initialize temperature profile
        self.init_T(planet, lat)
        
        # Initialize heat capacity profile
        self.update_cp()
        
        # Ice thermophysical properties
        rhod = 920. #density of ice at depth
        rhos = (1.-self.por)*rhod #density of ice at surface

        if (self.density == 'linear'):
            #Ice density increasing linearly with depth
            if (self.z_rho >= self.z[-1, 0]):
                #If z_rho is greater than the depth of the lowest ice model layer then the lowest model layer is where
                #the ice density reaches that of slab ice (rhod)
                self.rho = ( (rhod - rhos)/(self.z[-1, 0] - self.z[0, 0]) )*self.z + rhos

                rho = lambda x: ( (rhod - rhos)/(self.z[-1, 0] - self.z[0, 0]) )*x + rhos
                self.rho_ave = integrate.quad(rho, self.z[0, 0], self.z[-1, 0])[0]/(self.z[-1, 0] - self.z[0, 0])

            else:
                #Otherwise, z_rho is the depth at which the ice density reaches that of slab ice (rhod)
                self.rho = ( (rhod - rhos)/(self.z_rho - self.z[0, 0]) )*self.z + rhos
                self.rho[np.where(self.z >= self.z_rho)] = rhod

                rho1 = lambda x: ( (rhod - rhos)/(self.z_rho - self.z[0, 0]) )*x + rhos
                rho2 = lambda x: rhod
                self.rho_ave = ( integrate.quad(rho1, self.z[0, 0], self.z_rho)[0]+\
                           integrate.quad(rho2, self.z_rho, self.z[-1, 0])[0] )/(self.z[-1, 0] - self.z[0, 0])

        elif (self.density == 'exp'):
            #Ice density increasing exponetially with depth
            H = self.z_rho #e-folding scale of density with depth
            self.rho = rhod - (rhod-rhos)*np.exp(-self.z/H)*np.ones_like(self.z)

            rho = lambda x: rhod - (rhod-rhos)*np.exp(-x/H)
            self.rho_ave = integrate.quad(rho, self.z[0, 0], self.z[-1, 0])[0]/(self.z[-1, 0] - self.z[0, 0])

        else:
            #Ice density remaining constant at rhod with depth
            self.rho = rhod*np.ones_like(self.z)
            self.rho_ave = rhod

        self.kc = (1.578812338418425)*((2.5e-6)*self.rho**2 - (1.23e-4)*self.rho + 0.024)
        self.Gamma = np.sqrt(self.kc*self.rho*self.cp)
            
        # Initialize conductivity profile
        self.update_k()
        
        # Initialize CO2 mass balance
        self.dmdt = -np.ones(self.N_x) #change in CO2 areal mass density over time
        self.m_CO2 = np.zeros(self.N_x) #CO2 frost areal mass density [kg.m-2]
        
        # Initialize H2O ice mass balance
        self.m_sub = np.zeros(self.N_x) #this is the rate of change of areal H2O ice mass
        self.m_H2O = np.zeros(self.N_x)
        self.m_forced = np.zeros(self.N_x)
        self.m_free = np.zeros(self.N_x)
        
        self.m_mol = np.zeros(self.N_x)
        
        # Initialize Atmospheric Profile
        self.init_Atmo()
        
    # Initialize Atmospheric Profile (Based on Bapst et al. 2018)
    def init_Atmo(self):
        #Atmospheric Spatial Grid, non-uniform geometric
        u_star = self.u_wind*k_wind/np.log(z_vik/z0) #surface friction velocity
        self.P_atm = np.array([P_atm(self.Ls, self.elev)]) #Need inital atmospheric surface pressure to calculate viscosity
        visc = kineViscosity(self.T[0,0], self.P_atm[0]) #atmospheric kinematic viscosity
        self.d = 30*(visc/u_star) #thickness of near-surface laminar layer
        
        #Spatial grid extends from the surface at h=d to an altitude of h_max above the surface (default: 6 km)
        #Ensure that 'N_skin' grid points are within the first 'skin depth' (to balance accuracy with efficiency)
        Kappa = k_wind*u_star*self.d #Eddy diffusion coefficient of first atmospheric layer
        L = np.sqrt(Kappa*planets.Mars.day) #'Skin depth' of atmospheric model
        h1 = np.geomspace(self.d, L, num=self.N_skin, endpoint=False)
        h2 = np.geomspace(L, h_max, num=(self.N_atm-self.N_skin))
        self.h = np.append(h1, h2) #Two-tier geometric spatial grid
        
        #Wind profile ##This is not needed for the actual model to run##
        #self.u = (u_star/k_wind)*np.log(self.h/z0)

        #Limiting height
        f_coriolis = 2.*Omega*np.sin(self.lat) #coriolis parameter
        h_lim = 0.2*u_star/f_coriolis
        
        #Eddy diffusion coefficient
        self.Kappa = k_wind*u_star*self.h
        self.Kappa[np.where(self.h >= h_lim)] = self.Kappa[np.where(self.h >= h_lim)[0][0]]
        
        #Atmospheric temperature and pressure profiles
        self.T_atm = np.mean(self.T[0, :]) - 4.5e-3*self.h #atmospheric temperature profile
        self.T_atm[np.where(self.T_atm < 0)] = 0 #make sure temperatures can't be negative
        self.P_atm = P_atm(self.Ls, self.elev+(self.h*1e-3)) #atmospheric pressure profile
        
        #Atmospheric water vapor density profile
        e_vap = self.WaterVolMix(self.Ls, self.lat*180./np.pi) * self.P_atm * self.fH2O_obliq
        e_vap[np.where(e_vap <= 0)] = 0
        
        self.rho_vap = np.ones_like(self.h) * (e_vap*mw)/(Rg*self.T_atm) #Initialize profile with temperature profile
        
        #Atmospheric density profile
        self.rho_atm = ((self.P_atm*mc)/(Rg*self.T_atm)) - ((mc/mw)-1)*self.rho_vap #atmospheric density profile
        
        #Coefficients for vapor diffusion equation
        self.dh = np.diff(self.h) #layer thicknesses
        self.d3h = self.dh[1:]*self.dh[0:-1]*(self.dh[1:] + self.dh[0:-1])
        self.a = self.Kappa[0:-2]*2*self.dh[1:]/self.d3h[0:]
        self.b = self.Kappa[1:-1]*2*self.dh[0:-1]/self.d3h[0:]
        
    # Temperature initialization
    def init_T(self, planet=planets.Moon, lat=0):
        self.T = np.zeros([self.nlayers, self.N_x]) \
                 + T_eq(planet, lat)
    
    # Heat capacity initialization
    def update_cp(self):
        self.cp = 1600.*np.ones_like(self.T) #heatCapacityIce(self.planet, self.T)
    
    # Thermal conductivity initialization (temperature-dependent)
    def update_k(self):
        self.k = thermCond(self.kc, self.T)
    
    ##########################################################################
    ######################## Core Thermal Computation ########################
    ##########################################################################   
    @staticmethod
    @jit(nopython=True)
    def update_T_ice(dt, Qs, Qb, T, T_bl, T_atm, P_atm, g1, g2, k, cp, rho, rho_vap, rho_atm, rho_sat, emis, kc, dz, m_sub, SH,\
                     e_sat, x_dp, y_dp):
        #Update temperatures within ice profile where surface is clear of CO2 frost. Calculate surface and bottom temp via
        #energy balance. Use temperature formula from Hayne et al. (2017) to calculate temps in between.
        T_bl = 0.5*(T[0, :] + T_atm)
        SH = SensibleHeat(T[0, :], T_bl, T_atm, P_atm, rho_vap, rho_atm, e_sat, x_dp, y_dp)
        
        # Coefficients for temperature-derivative terms
        alpha = np.transpose(g1*k[0:-2, :].T)
        beta = np.transpose(g2*k[1:-1, :].T)
        
        # Temperature of first layer is determined by energy balance at the surface
        T[0, :] = surfTempIce(Qs, dt, T[0:3, :], SH, emis, kc, dz[0], m_sub)
        
        # Temperature of the last layer is determined by the interior heat flux
        T[-1, :] = botTemp(Qb, T, k[-2, :], dz[-1])
        
        # This is an efficient vectorized form of the temperature
        # formula, which is much faster than a for-loop over the layers
        T[1:-1, :] = T[1:-1, :] + dt/(rho[1:-1, :]*cp[1:-1, :]) * ( alpha*T[0:-2, :] - \
                       (alpha+beta)*T[1:-1, :] + beta*T[2:, :] )
        
        return T, T_bl
    
    @staticmethod
    @jit(nopython=True)
    def CO2Flux_ice(dt, Qs, Qb, T, T_bl, T_atm, P_atm, g1, g2, k, cp, rho, rho_vap, rho_atm, rho_sat, emis, kc, dz, m_sub, SH,\
                    e_sat, m_CO2, dmdt, x_dp, y_dp):
        #Update temperatures within ice profile where surface is covered in CO2 frost. Surface temp is set to CO2 frost temp.
        #Calculate bottom temp via energy balance. Use temperature formula from Hayne et al. (2017) to calculate temps in
        #between.
        T_bl = 0.5*(T[0, :] + T_atm)
        Ts = T[0, :]
        
        SH = SensibleHeat(T[0, :], T_bl, T_atm, P_atm, rho_vap, rho_atm, e_sat, x_dp, y_dp)
        
        x = emis_CO2*sigma*Ts**3
        y = 0.5*thermCond(kc, Ts)/dz[0]
        dmdt = (x*Ts - Qs - y*(-3*Ts+4*T[1, :]-T[2, :]) - SH - L_H2O*m_sub)/L_CO2
        
        # Update mass of CO2 frost
        m_CO2 = m_CO2 + dmdt*dt
        
        # Coefficients for temperature-derivative terms
        alpha = np.transpose(g1*k[0:-2, :].T)
        beta = np.transpose(g2*k[1:-1, :].T)
        
        # Temperature of the last layer is determined by the interior
        # heat flux
        T[-1, :] = botTemp(Qb, T, k[-2, :], dz[-1])
        
        # This is an efficient vectorized form of the temperature
        # formula, which is much faster than a for-loop over the layers
        T[1:-1, :] = T[1:-1, :] + dt/(rho[1:-1, :]*cp[1:-1, :]) * ( alpha*T[0:-2, :] - \
                       (alpha+beta)*T[1:-1, :] + beta*T[2:, :] )
        
        return T, T_bl, m_CO2, dmdt
    ##########################################################################
    ##########################################################################
    ##########################################################################

#---------------------------------------------------------------------------
"""

The functions defined below are used by the thermal code.

"""
#---------------------------------------------------------------------------

# Thermal skin depth [m]
# P = period (e.g., diurnal, seasonal)
# kappa = thermal diffusivity = k/(rho*cp) [m2.s-1]
def skinDepth(P, kappa):
    return np.sqrt(kappa*P/np.pi)

# The spatial grid is non-uniform, with layer thickness increasing downward
def spatialGrid(zs, m, n, b, N_x):
    dz = np.zeros([1, N_x]) + zs/m # thickness of uppermost model layer
    z = np.zeros([1, N_x]) # initialize depth array at zero
    zmax = zs*b # depth of deepest model layer

    i = 0
    while (np.any(z[i, :] < zmax)):
        i += 1
        h = dz[i-1, :]*(1+1/n) # geometrically increasing thickness
        dz = np.append(dz, [h], axis=0) # thickness of layer i
        z = np.append(z, [z[i-1, :] + dz[i, :]], axis=0) # depth of layer i
    
    return z

# Solar incidence angle-dependent albedo model
# A0 = albedo at zero solar incidence angle
# a, b = coefficients
# i = solar incidence angle
def albedoVar(A0, a, b, i):
    return A0 + a*(i/(np.pi/4))**3 + b*(i/(np.pi/2))**8

# Radiative equilibrium temperature at local noontime
def T_radeq(planet, lat):
    return ((1-planet.albedo)/(sigma*planet.emissivity) * planet.S * np.cos(lat))**0.25

# Equilibrium mean temperature for rapidly rotating bodies
def T_eq(planet, lat):
    return T_radeq(planet, lat)/np.sqrt(2)

# Heat capacity of regolith (temperature-dependent)
# This polynomial fit is based on data from Ledlow et al. (1992) and
# Hemingway et al. (1981), and is valid for T > ~10 K
# The formula yields *negative* (i.e. non-physical) values for T < 1.3 K
def heatCapacity(planet, T):
    c = planet.cpCoeff
    return np.polyval(c, T)

def heatCapacityIce(planet, T):
    c = planet.ice_cpCoeff
    return np.polyval(c, T)

# Temperature-dependent thermal conductivity
# Based on Mitchell and de Pater (1994) and Vasavada et al. (2012)
@jit(nopython=True)
def thermCond(kc, T):
    return kc*(1 + R350*T**3)

# Bottom layer temperature is calculated from the interior heat
# flux and the temperature of the layer above
@jit(nopython=True)
def botTemp(Qb, T, k, dz):
    T[-1, :] = T[-2, :] + (Qb/k)*dz
    return T[-1, :]

def getTimeStep(p, day):
    dt_min = np.min( F * p.rho[:-1, 0] * p.cp[:-1, 0] * p.dz**2 / p.k[:-1, 0] )
    return dt_min

@jit(nopython=True)
def satVaporPressureIce(T):
    #(Buck 1981) originally from (Wexler 1977)
    x = np.exp( -5865.3696/T + 22.241033 + 0.013749042*T - \
                      0.34031775e-4*T**2 + 0.26967687e-7*T**3 + 0.6918651*np.log(T) )
    return x

@jit(nopython=True)
def iceSubFree(D, visc, dp, drho, f_dp):
    #Ice sublimation due to free convection
    #(Dundas and Byrne 2010)
    
    #Convoluted way of taking the cbrt since np.cbrt is not numba supported
    x = ( dp * f_dp )
    y = np.sign(x) * (np.abs(x)) ** (1./3.)
    
    m_free = 0.14 * drho  * D * y
    
    return m_free

@jit(nopython=True)
def diffusionCoeff(T, P_atm):
    #Molecular diffusivity of water vapor in CO2 gas
    #(Dundas and Byrne 2010)
    D = 1.387e-5 * (T/273.15)**1.5 * (1e5/P_atm)
    return D

@jit(nopython=True)
def kineViscosity(T, P_atm):
    #Kinematic viscosity
    #(Dundas and Byrne 2010)
    x = 1.48e-5 * ((Rg*T)/(mc*P_atm)) * ((240.+293.15)/(240.+T)) * (T/293.15)**1.5
    return x

@jit(nopython=True)
def densityRatio(T_atm, P_atm, rho_vap, e_sat, x_dp, y_dp):
    #ratio of the density difference between saturated and background atmospheric gases to an atmospheric density
    #(Dundas and Byrne 2010)
    e_vap = (rho_vap*Rg*T_atm)/mw
    
    x = x_dp - (mc - mw)*e_vap/T_atm
    y = y_dp - 0.5*(mc - mw)*e_vap/T_atm
    
    ratio = x/y
    ratio[np.where(x/y < 0)] = 0
    
    print ('\n densityRatio:', ratio)
    
    return ratio

@jit(nopython=True)
def densityRatio1(drho, rho_surf):
    #ratio of the density difference between saturated and background atmospheric gases to an atmospheric density
    #(Dundas and Byrne 2010)
    
    #ratio = ( rho_atm - rho_surf )/rho_surf
    ratio = ( ((mc/mw)-1.)*(-drho) )/rho_surf
    ratio[np.where(ratio < 0)] = 0
    
    print ('densityRatio1:', ratio)
    
    return ratio

# Surface temperature calculation using Newton's root-finding method
@jit(nopython=True)
def surfTempIce(Qs, dt, T, SH, emis, kc, dz, m_sub):
    #(Qs, dt, self.T[0:3], self.SH, self.emissivity, self.kc[0], self.dz[0], self.m_sub)
    #Ts = T[0, :]
    deltaT = T[0, :]
    
    while (np.any(np.abs(deltaT) > DTSURF)):
        x = emis*sigma*T[0, :]**3
        y = 0.5*thermCond(kc, T[0, :])/dz
    
        # f is the function whose zeros we seek
        f = x*T[0, :] - Qs - y*(-3*T[0, :]+4*T[1, :]-T[2, :]) - SH - L_H2O*m_sub
        # fp is the first derivative w.r.t. temperature        
        fp = 4*x - \
             3*kc*R350*T[0, :]**2 * \
                0.5*(4*T[1, :]-3*T[0, :]-T[2, :])/dz + 3*y
        
        # Estimate of the temperature increment
        deltaT = -f/fp
        T[0, :] += deltaT
        #print ('deltaT:', deltaT)
        #print ('Ts:', Ts, '\n')
    # Update surface temperature
    return T[0, :]#, SH, emis, kc, m_sub
    
def MarsOpticalDepth():
    #Read optical depth data from Mars Climate Database
    df = pd.read_csv('MarsOpticalDepths_60E.txt', delim_whitespace=True, skiprows=10, header=0, names=np.arange(0, 50))
    #Reformat data into a usable one
    df = df.drop(df.index[1])
    df = df.drop(1, axis=1)
    df.columns = np.arange(0, df.shape[1])
    df = df.reset_index(drop=True)
    lat = df.values[0,1:]
    Ls = df.values[1:,0]
    Ls = [float(i) for i in Ls]
    dfn = df.values[1:,1:]
    #Create interpolation function that will return an optical depth for any given lat and Ls (nu)
    x, y = np.meshgrid(Ls, lat)
    f = interp2d(Ls, lat, np.transpose(dfn), kind='cubic')
    return f

def MarsAtmTemps():
    #Read atmospheric temperature data from Mars Climate Database
    df = pd.read_csv('MarsAtmTemps.txt', delim_whitespace=True, skiprows=10, header=0, names=np.arange(0, 50))
    #Reformat data into a usable one
    df = df.drop(df.index[1])
    df = df.drop(1, axis=1)
    df.columns = np.arange(0, df.shape[1])
    df = df.reset_index(drop=True)
    lat = df.values[0,1:]
    Ls = df.values[1:,0]
    Ls = [float(i) for i in Ls]
    dfn = df.values[1:,1:]
    #Create interpolation function that will return an atmospheric temperature for any given lat and Ls (nu)
    x, y = np.meshgrid(Ls, lat)
    f = interp2d(Ls, lat, np.transpose(dfn), kind='cubic')
    return f

def MarsWaterVolumeMixingRatio():
    #Read water volume mixing ratio data from Mars Climate Database [mol.mol-1]
    df = pd.read_csv('MarsWaterVolMixing_45E_6km.txt', delim_whitespace=True, skiprows=10, header=0, names=np.arange(0, 50))
    #Reformat data into a usable one
    df = df.drop(df.index[1])
    df = df.drop(1, axis=1)
    df.columns = np.arange(0, df.shape[1])
    df = df.reset_index(drop=True)
    lat = df.values[0,1:]
    Ls = df.values[1:,0]
    Ls = [float(i) for i in Ls]
    dfn = df.values[1:,1:]
    #Create interpolation function that will return a water volume mixing ratio for any given lat and Ls (nu)
    x, y = np.meshgrid(Ls, lat)
    f = interp2d(Ls, lat, np.transpose(dfn), kind='cubic')
    return f
@jit
def P_vik(S, SR=360.50865):
    #Viking 2 Lander Pressure Curve as a function of martian sol (Tillman et al. 1993)
    SYR = 668.59692
    P0 = 8.66344
    Pi = np.array([0.798, 0.613, 0.114, 0.063, 0.018])
    phi = np.array([93.59, -131.37, -67.50, 17.19, 98.84])
    
    P = P0*np.ones_like(S)
    
    for i in range(0, Pi.size):
        P = P + Pi[i] * np.sin( 2.*np.pi*(i+1)*(S-SR)/SYR + (phi[i]*np.pi/180.) )
    
    return P * 100 #convert mbar to Pa
@jit
def MJD(Ls, n=0):
    #Conversion from Ls to modified Julian date
    #n is the nth orbit of Mars since the epoch 1874.0
    #(Allison and McEwen 2000 Eq. 14)
    mjd = 51507.5 + 1.90826*(Ls-251) - 20.42*np.sin((Ls-251)*np.pi/180.) + 0.72*np.sin(2.*(Ls-251)*np.pi/180.) + \
                ( 686.9726 + 0.0043*np.cos((Ls-251)*np.pi/180.) - 0.0003*np.cos(2.*(Ls-251)*np.pi/180.) )*(n-66)
    return mjd
@jit
def MSD(mjd):
    #Conversion from modified Julian date to Martian Sol Date
    #(Allison and McEwen 2000 Eq. 32)
    k = 0.001
    msd = ( (mjd - 51549.0)/1.02749125 ) + 44796.0 - k - 395.77492432270083
    return msd
@jit
def P_atm(Ls, elev=-4.23):
    #Atmospheric pressure at a given Ls [degrees] and elevation [km] based on Viking 2 Lander data and hydrostatic assumption
    mjd = MJD(Ls)
    msd = MSD(mjd)
    Pv = P_vik(msd)
    P0 = Pv/np.exp(-(-4.23)/H_atm)
    
    P = P0*np.exp(-elev/H_atm)
    return P

def MarsOrbitalParams(t):
    #mars eccentricity, obliquity, and longitude of perihelion at time t in kyr (Earth years!) relative to present
    A = np.load('La2004.npy') #Use La2004 orbital solutions
    idx = np.where( t==A[:,0] )
    ecc = np.asscalar(A[idx,1]) #eccentricity
    obliq = np.asscalar(A[idx,2]) #obliquity [rad]
    Lp = np.asscalar(A[idx,3]) + np.pi #longitude of perihelion [rad]
                                       #(for some reason the La2004 Lp values are 180 degrees off?)
    if (Lp >= 2.*np.pi):
        Lp = Lp - 2.*np.pi
    return ecc, obliq, Lp

def partialPressureObliq(obliq):
    #partial pressure of water vapor in Martian atmosphere as it depends on obliquity
    #(Schorghofer and Forget 2012) (used in Bramson 2017)
    #obliquity in degrees
    a = -1.27
    b = 0.139
    c = -0.00389
    if (obliq >= 10. and obliq <= 28.):
        P = np.exp(a + b*(obliq - 28))
    elif (obliq > 28. and obliq <= 50.):
        P = np.exp(a + b*(obliq - 28) + c*(obliq - 28)**2)
    return P

@jit(nopython=True)
def SensibleHeat(Ts, T_bl, T_atm, P_atm, rho_vap, rho_atm, e_sat, x_dp, y_dp):
    # From Dundas and Byrne 2010, also used in Williams et al. 2008
    dp = densityRatio(T_atm, P_atm, rho_vap, e_sat, x_dp, y_dp)
    visc = kineViscosity(T_bl, P_atm)
    dyn_visc = rho_atm*visc
    
    #SH_forced = p.rho_atm[0] * cp_atm * A_drag * p.u_wind * (T_atm - Ts)
    
    x = ( (cp_atm*dyn_visc/k_atm) * (g_mars/(visc**2)) * dp )
    y = np.sign(x) * (np.abs(x)) ** (1./3.)
    SH_free = 0.14 * (T_atm - Ts) * k_atm * y
    
    SH = SH_free #SH_forced + SH_free
    
    return SH

def viewFactors(p):
    #First do ray tracing to calculate index of all other surface points seen by each individual surface point
    los_idx = []
    F_los = []
    idx_max = np.zeros(p.x.shape[1])
    idx_min = np.zeros(p.x.shape[1])
    
    idx2 = np.argmax( (p.x[1,0] -  p.x[1,1:])/(p.x[0,0] -  p.x[0,1:]) )
    idx4 =  np.array([ i for i in range(1,idx2+2) if np.all( (p.x[1,0] - p.x[1,i])/(p.x[0,0] - p.x[0,i]) >= \
                                                      (p.x[1,0] - p.x[1,1:i])/(p.x[0,0] - p.x[0,1:i]) ) ])
    
    idx_max[0] = np.amax(idx4)
    idx_min[0] = 0
    
    #Calculate view factor for first point
    dx = p.dx #p.x[0,1] - p.x[0,0]
    A = dx**2
    
    R = np.sqrt( ( p.x[0,0] - p.x[0,idx4] )**2 + ( p.x[1,0] - p.x[1,idx4] )**2 )
    alpha_12 = np.arctan((p.x[1,0] -  p.x[1,idx4])/(p.x[0,0] -  p.x[0,idx4]))
    
    #if/else statement used to evaluate difference between 2D (xz-plane) and 2.5D (extended y-axis) surface evolution
    if (p.yaxis_ext == True):
        #First factor in expression below assumes surface extends +/-infinitely along y-axis (assuming surface in x-z plane)
        F = (np.pi*R/(2.*dx)) * (A/(np.pi*R**2)) * np.sin( np.abs(p.alpha[0]) + (idx4/np.abs(idx4))*alpha_12 ) \
                         * np.sin( np.abs(p.alpha[idx4]) - (idx4/np.abs(idx4))*alpha_12 )
    else:
        F = (A/(np.pi*R**2)) * np.sin( np.abs(p.alpha[0]) + (idx4/np.abs(idx4))*alpha_12 ) \
                         * np.sin( np.abs(p.alpha[idx4]) - (idx4/np.abs(idx4))*alpha_12 )
    
    idx4 = idx4[np.where( F >= 0 )] #Get rid of indicies where view factor F is negative
    F = F[np.where( F >= 0 )]
    
    los_idx.append(idx4)
    F_los.append(F)
    
    for idx in range(1,p.N_x-1):
        idx1 = np.argmin( (p.x[1,idx] - p.x[1,0:idx])/(p.x[0,idx] - p.x[0,0:idx]) )
        idx2 = np.argmax( (p.x[1,idx] - p.x[1,idx+1:])/(p.x[0,idx] - p.x[0,idx+1:]) )
        
        idx3 = [ i for i in range(1, idx-idx1+1) if np.all( (p.x[1,idx] - p.x[1,idx-i])/(p.x[0,idx] - p.x[0,idx-i]) <= \
                                                           (p.x[1,idx] - p.x[1,idx-i:idx])/(p.x[0,idx] - p.x[0,idx-i:idx]) ) ]
        idx4 = [ i for i in range(1,idx2+2) if np.all( (p.x[1,idx] - p.x[1,idx+i])/(p.x[0,idx] - p.x[0,idx+i]) >= \
                                                      (p.x[1,idx] - p.x[1,idx+1:idx+i])/(p.x[0,idx] - p.x[0,idx+1:idx+i]) ) ]   
        
        idx5 = np.append(-np.array(idx3), np.array(idx4))
        
        idx_max[idx] = np.amax(idx5)
        idx_min[idx] = np.amin(idx5)
        
        R = np.sqrt( ( p.x[0,idx] - p.x[0,idx+idx5] )**2 + ( p.x[1,idx] - p.x[1,idx+idx5] )**2 )
        alpha_12 = np.arctan((p.x[1,idx] -  p.x[1,idx+idx5])/(p.x[0,idx] -  p.x[0,idx+idx5]))
        
        if (p.yaxis_ext == True):
            F = (np.pi*R/(2.*dx)) * (A/(np.pi*R**2)) * np.sin( np.abs(p.alpha[idx]) + (idx5/np.abs(idx5))*alpha_12 ) \
                             * np.sin( np.abs(p.alpha[idx+idx5]) - (idx5/np.abs(idx5))*alpha_12 )
        else:
            F = (A/(np.pi*R**2)) * np.sin( np.abs(p.alpha[idx]) + (idx5/np.abs(idx5))*alpha_12 ) \
                             * np.sin( np.abs(p.alpha[idx+idx5]) - (idx5/np.abs(idx5))*alpha_12 )
        
        idx5 = idx5[np.where( F >= 0 )] #Get rid of indicies where view factor F is negative
        F = F[np.where( F >= 0 )]
        
        
        los_idx.append(idx5)
        F_los.append(F)
    
    idx1 = np.argmin( (p.x[1,p.N_x-1] - p.x[1,0:p.N_x-1])/(p.x[0,p.N_x-1] - p.x[0,0:p.N_x-1]) )
    idx3 = -np.array([ i for i in range(1, p.N_x-1-idx1+1) if np.all( (p.x[1,p.N_x-1] - p.x[1,p.N_x-1-i])/ \
                                                           (p.x[0,p.N_x-1] - p.x[0,p.N_x-1-i]) <= \
                                                           (p.x[1,p.N_x-1] - p.x[1,p.N_x-1-i:p.N_x-1])/ \
                                                           (p.x[0,p.N_x-1] - p.x[0,p.N_x-1-i:p.N_x-1]) ) ])
    
    idx_min[p.N_x-1] = np.amin(idx3)
    idx_max[p.N_x-1] = 0
    
    R = np.sqrt( ( p.x[0,p.N_x-1] - p.x[0,p.N_x-1+idx3] )**2 + ( p.x[1,p.N_x-1] - p.x[1,p.N_x-1+idx3] )**2 )
    alpha_12 = np.arctan((p.x[1,p.N_x-1] -  p.x[1,p.N_x-1+idx3])/(p.x[0,p.N_x-1] -  p.x[0,p.N_x-1+idx3]))

    if (p.yaxis_ext == True):
        F = (np.pi*R/(2.*dx)) * (A/(np.pi*R**2)) * np.sin( np.abs(p.alpha[p.N_x-1]) + (idx3/np.abs(idx3))*alpha_12 ) \
                         * np.sin( np.abs(p.alpha[p.N_x-1+idx3]) - (idx3/np.abs(idx3))*alpha_12 )
    else:
        F = (A/(np.pi*R**2)) * np.sin( np.abs(p.alpha[p.N_x-1]) + (idx3/np.abs(idx3))*alpha_12 ) \
                         * np.sin( np.abs(p.alpha[p.N_x-1+idx3]) - (idx3/np.abs(idx3))*alpha_12 )

    idx3 = idx3[np.where( F >= 0 )] #Get rid of indicies where view factor F is negative
    F = F[np.where( F >= 0 )]
    
    los_idx.append(idx3)
    F_los.append(F)
    
    return F_los, los_idx, idx_max.astype(int), idx_min.astype(int)

#@jit(nopython=True)
def groundFlux(F_los, los_idx, T, emissivity, Q_solar, Albedo, N_x, Q_surf_vis, Q_surf_IR, Q_surf):
    #Reradiated IR flux and reflected solar (visible) flux from surfaces within line of sight of each surface
    #Q_surf_vis = np.zeros(N_x)
    #Q_surf_IR = np.zeros(N_x)
    #Q_surf = np.zeros(N_x)
    for i in range(0, N_x):
        idx = los_idx[i]
        #Include both the IR and visible flux
        Q_surf_vis[i] = (1. - Albedo[i]) * np.sum( F_los[i] * (Albedo[i+idx]/(1.-Albedo[i+idx])) * Q_solar[i+idx] )
        Q_surf_IR[i] = np.sum( F_los[i] * emissivity[i+idx] * sigma * T[i+idx]**4 )
        Q_surf[i] = Q_surf_vis[i] + Q_surf_IR[i]
    
    return Q_surf, Q_surf_vis, Q_surf_IR

@jit(nopython=True)
def Shadowfax(r, sa, idx_max, idx_min, cosz):
    #Determine which points are shadowed by surrounding terrain.
    #This assumes that the topography of the 2D surface extends linearly in the 3rd dimension.
    #Shows the meaning of haste
    x = r[0,:]
    y = r[1,:]
    
    n = np.arange(0,idx_max.size)
    theta = np.zeros(idx_max.size)
    
    if (sa >= np.pi/2. and sa <= 3.*np.pi/2.):
        h = (y[n+idx_max] - y[:])
        d = (x[n+idx_max] - x[:])
        phi = sa - np.pi
        
        #Don't include points on edges. Assume no shadowing from edgeward of these points.
        theta[0:-1] = np.arctan(h[0:-1]*np.cos(phi)/d[0:-1])
    
    elif (sa < np.pi/2.):
        h = (y[n+idx_min] - y[:])
        d = (x[n+idx_min] - x[:])
        phi = sa
        
        #Don't include points on edges. Assume no shadowing from edgeward of these points.
        theta[1:] = -np.arctan(h[1:]*np.cos(phi)/d[1:])
    
    elif (sa > 3.*np.pi/2.):
        h = (y[n+idx_min] - y[:])
        d = (x[n+idx_min] - x[:])
        phi = sa - 2.*np.pi
        
        #Don't include points on edges. Assume no shadowing from edgeward of these points.
        theta[1:] = -np.arctan(h[1:]*np.cos(phi)/d[1:])
    
    idx_shadow = np.where( theta > np.pi/2. - np.arccos(cosz) )
    
    return idx_shadow, theta

@jit(nopython=True)
def iceSubMolecularDiffusion(D, d, drho):
    #Ice mass loss due to molecular diffusion through laminar layer [kg.m-2.s-1]
    #D = diffusionCoeff(T_bl, P_atm)
    #m_mol = (D/d) * (rho_vap-rho_sat)
    
    m_mol = (D/d) * drho
    
    return m_mol

@jit(nopython=True)
def surfVaporDensity(dt, m_H2O, m_sub, m_mol, m_free, d, dh, rho_vap, rho_sat, Kappa, c_adv, visc, D, rho_surf, drho, f_dp):
    #Calculate water vapor density at lower boundary of atmosphere
    
    # no need for these (e_sat, x_dp, y_dp, P_atm, T_atm, rho_atm)
    
    #Ice mass flux
    #dp = densityRatio(T_atm, P_atm, rho_vap[0], e_sat, x_dp, y_dp)
    drho = rho_vap[0] - rho_sat
    #y_dp = rho_vap[0] - rho_sat
    dp = densityRatio1(drho, rho_surf)
    
    if (np.all(dp < 5e-6)):
        m_free[:] = 0
    else:
        m_free = iceSubFree(D, visc, dp, drho, f_dp)
    
    #m_free = iceSubFree(D, visc, dp, drho, f_dp)
    
    #m_free = iceSubFree(T_atm, P_atm, rho_vap[0], rho_sat, D, visc, e_sat, x_dp, y_dp)
    m_mol = iceSubMolecularDiffusion(D, d, drho)
    m_sub = m_mol + m_free
    m_H2O = m_H2O + m_sub*dt
    
    #Diffused quantity is now water vapor density as opposed to water mixing ratio as above
    #Advection term is now a second order approximation of the first derivative
    a0 = -2.*dh[0]*dh[1] - dh[1]**2
    a1 = (dh[0] + dh[1])**2
    a2 = -(dh[0]**2)
    b0 = 2.*dh[1]
    b1 = -2.*(dh[0] + dh[1])
    b2 = 2.*dh[0]
    dh3 = (dh[0]*dh[1]*(dh[0]+dh[1]))
    
    rho_vap[0] = rho_vap[0] + dt*( ( (a0*Kappa[0]+a1*Kappa[1]+a2*Kappa[2])/dh3 ) \
                                    * ( (a0*rho_vap[0]+a1*rho_vap[1]+a2*rho_vap[2])/dh3 ) \
                                    + Kappa[0]*( (b0*rho_vap[0]+b1*rho_vap[1]+b2*rho_vap[2])/dh3 ) ) \
                                + dt*c_adv[0]*( (a0*rho_vap[0]+a1*rho_vap[1]+a2*rho_vap[2])/dh3 ) \
                                + (dt/dh[0])*(-np.mean(m_sub))
    
    #Ensure that there are no negative vapor density values
    if (rho_vap[0] <= 0):
        rho_vap[0] = 0
    
    return rho_vap, m_H2O, m_sub, m_mol, m_free, drho

@jit(nopython=True)
def upperVaporDensity(e_vap, T_atm, rho_vap):
    #Calculate water vapor partial pressure at upper boundary of atmospheric model ensuring no negative values
    #Using water volume (molar) mixing ratio as mole fraction. The two are ~equal since H2O is a minor atmospheric constituent
    
    rho_vap[-1] = (e_vap[0]*mw)/(Rg*T_atm[-1])

@jit(nopython=True)
#Solve diffusion equations for atmospheric water vapor profile
def vapor_density_profile(dt, m_H2O, m_sub, m_mol, m_free, T_atm, P_atm, d, dh, rho_vap, rho_atm, rho_sat, \
                          e_vap, Kappa, a, b, cbrt, visc, D, D_atm, visc_atm, rho_surf, drho, f_dp, vap_set):
    
    #no need for these (x_dp, y_dp, e_sat)
    
    #Atmospheric density profile with water vapor contribution
    rho_atm = ((P_atm*mc)/(Rg*T_atm)) - ((mc/mw)-1)*rho_vap
    #Ensure atmospheric density can never drop below water vapor density
    rho_atm[np.where(rho_atm <= rho_vap)] = rho_vap[np.where(rho_atm <= rho_vap)]
    
    #Useful coefficients
    x = (rho_vap[1:-1] + rho_vap[0:-2])/(rho_atm[1:-1] + rho_atm[0:-2])
    y = (rho_vap[2:] + rho_vap[1:-1])/(rho_atm[2:] + rho_atm[1:-1])

    #Advection rate parameter (free convection) (after Bapst et al. 2018)
    #D_atm = diffusionCoeff(T_atm[:-1], P_atm[:-1])
    #g = g_mars
    #visc_atm = kineViscosity(T_atm[:-1], P_atm[:-1])
    dp = rho_atm[1:] - rho_atm[0:-1]
    
    #np.cbrt( (g/visc)*(dp/dh)*(D**2/rho_atm[0:-1])*(dh**4) , cbrt)
    #Convoluted way of taking the cbrt since np.cbrt is not numba supported
    cbrt = ((g_mars/visc_atm)*(dp/dh)*(D_atm**2/rho_atm[0:-1])*(dh**4))
    c_adv = 0.15 * (np.sign(cbrt) * (np.abs(cbrt)) ** (1./3.)) / dh
    c_adv[np.where( c_adv <= 0 )] = 0 #Free convection only causes mixing in vertical direction
    
    #Get vapor density at upper boundary
    #rho_vap = upperVaporDensity(e_vap, T_atm[-1], rho_vap)
    if (vap_set == 1):
        rho_vap[-1] = (e_vap[0]*mw)/(Rg*T_atm[-1])
    else:
        rho_vap[-1] = vap_set
    
    #Solve for vapor density at surface
    rho_vap, m_H2O, m_sub, m_mol, m_free, drho = surfVaporDensity(dt, m_H2O, m_sub, m_mol, m_free, d, dh, rho_vap, rho_sat,\
                                                                  Kappa, c_adv, visc, D, rho_surf, drho, f_dp)
    
    
    #Vectorized form of the water vapor diffusion + advection equation
    #self.rho_vap[1:-1] = self.rho_vap[1:-1] + dt*( self.a*self.rho_vap[0:-2] - (self.a+self.b)*self.rho_vap[1:-1] + \
    #                                              self.b*self.rho_vap[2:] + self.a*x*self.rho_atm[0:-2] - \
    #                                              (self.a*x+self.b*y)*self.rho_atm[1:-1] + self.b*y*self.rho_atm[2:] ) \
    #                                        + (dt*self.c_adv[1:]/self.dh[1:])*( self.rho_vap[2:] - self.rho_vap[1:-1] )

    #Changed several plus/minus signs based on notes. Original notation may have been in error.
    rho_vap[1:-1] = rho_vap[1:-1] + dt*( a*rho_vap[0:-2] - (a+b)*rho_vap[1:-1] + \
                                                  b*rho_vap[2:] - a*x*rho_atm[0:-2] + \
                                                  (a*x+b*y)*rho_atm[1:-1] - b*y*rho_atm[2:] ) \
                                            + (dt*c_adv[1:]/dh[1:])*( rho_vap[2:] - rho_vap[1:-1] )

    ##Diffused quantity is now water vapor density as opposed to water mixing ratio as above
    ##Advection term is now a second order approximation of the first derivative
    #a0 = -(self.dh[1:]**2)
    #a1 = self.dh[1:]**2 - self.dh[0:-1]**2
    #a2 = self.dh[0:-1]**2
    #b0 = 2.*self.dh[1:]
    #b1 = -2.*(self.dh[0:-1] + self.dh[1:])
    #b2 = 2.*self.dh[0:-1]
    #dh3 = self.dh[0:-1]*self.dh[1:]*(self.dh[0:-1]+self.dh[1:])
    #
    #self.rho_vap[1:-1] = self.rho_vap[1:-1] + dt*( ( (a0*self.Kappa[0:-2]+a1*self.Kappa[1:-1]+a2*self.Kappa[2:])/dh3 ) \
    #                         * ( (a0*self.rho_vap[0:-2]+a1*self.rho_vap[1:-1]+a2*self.rho_vap[2:])/dh3 )\
    #                         + self.Kappa[1:-1]*( (b0*self.rho_vap[0:-2]+b1*self.rho_vap[1:-1]+b2*self.rho_vap[2:])/dh3 ) )\
    #                     + dt*self.c_adv[1:]*( (a0*self.rho_vap[0:-2]+a1*self.rho_vap[1:-1]+a2*self.rho_vap[2:])/dh3 )
    #
    #self.diff1 = dt*( ( (a0*self.Kappa[0:-2]+a1*self.Kappa[1:-1]+a2*self.Kappa[2:])/dh3 ) \
    #                         * ( (a0*self.rho_vap[0:-2]+a1*self.rho_vap[1:-1]+a2*self.rho_vap[2:])/dh3 ) )
    #self.diff2 = dt*( self.Kappa[1:-1]*( (b0*self.rho_vap[0:-2]+b1*self.rho_vap[1:-1]+b2*self.rho_vap[2:])/dh3 ) )
    #self.adve1 = dt*self.c_adv[1:]*( (a0*self.rho_vap[0:-2]+a1*self.rho_vap[1:-1]+a2*self.rho_vap[2:])/dh3 )

    #self.adve2[0] = -self.adve
    #self.adve2[1:] = -dt*self.c_adv[1:-1]*( (a0*self.rho_vap[0:-3]+a1*self.rho_vap[1:-2]+a2*self.rho_vap[2:-1])/dh3 )

    ##Also need to add advected water vapor to layer above (?)
    #self.rho_vap[1] = self.rho_vap[1] - self.adve #The second atmospheric layer adds advected vapor from the first layer
    #self.rho_vap[2:-1] = self.rho_vap[2:-1] - dt*self.c_adv[1:-1]\
    #                                           *( (a0*self.rho_vap[0:-3]+a1*self.rho_vap[1:-2]+a2*self.rho_vap[2:-1])/dh3 )

    #Ensure no negative values of water vapor density
    rho_vap[np.where(rho_vap < 0)] = 0
    
    return (dt, m_H2O, m_sub, m_mol, m_free, T_atm, P_atm, d, dh, rho_vap, rho_atm, rho_sat, e_vap, Kappa, a, b, cbrt, drho)
    
def getTimeStepAtmo(p, dt):
    #Get timestep for stability in atmospheric model
    D = diffusionCoeff(p.T[0, 0], p.P_atm[0])
    dt_atm = p.dh**2/(2.*p.Kappa[:-1]) #p.dh**2/(2.*p.Kappa[:-1])
    dt_atm = np.append( dt_atm, [(p.d*p.dh[0])/(2.*D)] )
    #print ('dt_atm:', dt_atm)
    dt_atm = np.amin(dt_atm)
    #Make atmospheric timestep an integer factor of thermal timestep
    n = np.ceil(dt/dt_atm)
    dt_atm = dt/n
    return dt_atm

@jit(nopython=True)
def updateOrbitAtmo(dt_atm, rAU, ecc, nu_atm, obliq, r_atm, nudot_atm):
    nudot_atm = orbitParamsAtmo(rAU, ecc, nu_atm, obliq, r_atm, nudot_atm)
    nu_atm += nudot_atm * dt_atm
    return nu_atm, nudot_atm

@jit(nopython=True)
def orbitParamsAtmo(rAU, ecc, nu_atm, obliq, r_atm, nudot_atm):

    # Useful parameter:
    x = rAU*(1 - ecc**2)

    # Distance to Sun
    r_atm = x/(1 + ecc*np.cos(nu_atm))

    # Angular velocity
    nudot_atm = r_atm**-2 * np.sqrt(GM*x)
    return nudot_atm

@jit(nopython=True)
def hourAngle(t, P):
    return (2.*np.pi * t/P) % (2.*np.pi)

@jit(nopython=True)
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
    
    #cosine of solar azimuth above slope
    #if statement ensures that polar night will be observed
    if (cosz <= 0):
        cos_slope = np.zeros_like(alpha)
    else:
        cos_slope = np.cos(alpha)*cosz + np.sin(alpha)*sinz*np.cos(a)
    
    #Clipping function = zero when sun is below local slope horizon
    y = 0.5*(cos_slope + np.abs(cos_slope))
    
    return y, sa

@jit(nopython=True)
def cosSolarZenith(lat, dec, h):
    
    # Cosine of solar zenith angle
    x = np.sin(lat)*np.sin(dec) + np.cos(lat)*np.cos(dec)*np.cos(h)
    
    # Clipping function = zero when sun below horizon:
    y = 0.5*(x + np.abs(x))
    
    return y

def init_Surface(dx, scale, length, numtimes=int(1e4), P0=6.5e-4):
    #Initialize surface with uniform power distribution across all wavelengths
    x = np.arange(0, length + dx, dx)
    k = np.fft.fftfreq(x.shape[-1], d=dx)
    #P0 = 6.5e-4
    P = P0*np.ones_like(k)

    ys = np.zeros([k.size, numtimes+1])
    yh = np.zeros([k.size, numtimes+1])
    FTs = np.zeros([k.size, numtimes+1])
    FTh = np.zeros([k.size, numtimes+1])

    FTs[:,0] = np.sqrt(P)
    ys[:,0] = np.fft.ifft(FTs[:,0])
    np.random.seed()
    yh[:,0] = np.random.normal(scale=scale, size=k.size)
    FTh[:,0] = np.fft.fft(yh[:,0])
    
    for i in range(1, numtimes+1):
        FTs[:,i] = FTh[:,i-1] * ( np.abs(FTs[:,i-1]) / np.abs(FTh[:,i-1]) )
        ys[:,i] = np.fft.ifft(FTs[:,i])

        idx_s = np.argsort(ys[:,i])
        yh[idx_s,i] = np.sort(yh[:,i-1])
        FTh[:,i] = np.fft.fft(yh[:,i])
    
    yh[:,-1] = yh[:,-1] - np.mean(yh[:,-1])
    ys[:,-1] = ys[:,-1] - np.mean(ys[:,-1])
    
    y = ys[:,-1]

    alpha = np.zeros_like(x)

    alpha[0] = np.arctan((y[1] - y[0])/(x[1] - x[0]))
    alpha[-1] = np.arctan((y[-1] - y[-2])/(x[-1] - x[-2]))
    alpha[1:-1] = (np.arctan( (y[1:-1] - y[0:-2])/(x[1:-1] - x[0:-2]) ) + \
                   np.arctan( (y[2:] - y[1:-1])/(x[2:] - x[1:-1]) )) /2.
    
    return x, y, alpha

def runSurfaceEvolution(n, x0, y0, alpha0, scale, N_runs=0, run_path='', run_num=0, dx=0.1, f_year=100., planet=planets.Mars,\
                        lat=(85.*(np.pi/180)), ndays=668.59692, Ls=0., beta=0, f_IR=0.04, f_scat=0.02, u_wind=3., b_exp=0.2,\
                        atmExt=True, elev=-3., t_start=-1, dt=0, A_ice=0.35, shadow=True, yaxis_ext=True, por=0.5, \
                        z_rho=10., density='linear', N_atm=16, N_skin=7, Ls_eq=0.):
    # Interpolated Version Surface Evolution Function
    #n: Number of surfaces in evolved surface array. The number of evolution models run will be n-1.
    #x0, y0, alpha: length, height, and slope arrays of initial surface
    #scale: standard deviation of gaussian used to initialize vertial surface roughness
    #dx: horizontal resolution of points making up surface
    #f_year: Evolution timestep. Number of years (or fraction of year) over which mass balance is estimated to be unchanged.
    start = time.time()
    
    rho_ice = 920.

    x = np.zeros([x0.size, n])
    y = np.zeros([y0.size, n])
    alpha = np.zeros([alpha0.size, n])

    m_H2O = np.zeros([x0.size, n-1])
    m_CO2_max = np.zeros([x0.size, n-1])
    T_max = np.zeros([x0.size, n-1])

    x[:,0] = x0
    y[:,0] = y0
    alpha[:,0] = alpha0
    
    #Make new directory or update existing directory
    if (N_runs == 0):
        path = 'atmo5_2D_d'+str(np.amax(x[:,0]))+'m_R'+str(dx)+'m_VE'+str(scale)+'m_n'+str(n)+'_f'+str(f_year)+'yr_tstart'\
                               +str(t_start)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
    else:
        subpath = '/run_' + str(run_num)
        if os.path.exists(run_path+subpath):
            shutil.rmtree(run_path+subpath)
        os.mkdir(run_path+subpath)
    
    t = t_start #Time in kyr from present. Ex: t=-1 == 1000yr before present.
    t_obliq = t
    
    for i in range(1, n):
        print ('\ni:', i)
        x1 = np.array([x[:,i-1],y[:,i-1]])
        alpha1 = alpha[:,i-1]
        
        mars = model(planet=planet, lat=lat, ndays=ndays, Ls=Ls, x=x1, alpha=alpha1, beta=beta, \
                                         f_IR=f_IR, f_scat=f_scat, u_wind=u_wind, b_exp=b_exp, \
                                         atmExt=atmExt, elev=elev, t_obliq=t_obliq, dt=dt, A_ice=A_ice, dx=dx, \
                                         shadow=shadow, yaxis_ext=yaxis_ext, por=por, z_rho=z_rho, \
                                         density=density, N_atm=N_atm, N_skin=N_skin, Ls_eq=Ls_eq)
        mars.run()
        
        m_H2O[:,i-1] = mars.m_H2Ot[-1,:]
        m_CO2_max[:,i-1] = np.amax(mars.m_CO2t[:,:], axis=0)
        T_max[:,i-1] = np.amax(mars.T[:,0,:], axis=0)

        x[:,i] = x[:,i-1] + f_year*(m_H2O[:,i-1]/(mars.profile.rho_ave))*(-np.sin(alpha1*np.pi/180.))
        y[:,i] = y[:,i-1] + f_year*(m_H2O[:,i-1]/(mars.profile.rho_ave))*(np.cos(alpha1*np.pi/180.))

        f = interp1d(x[:,i], y[:,i], kind='cubic', fill_value='extrapolate')

        x[:,i] = x0
        y[:,i] = f(x[:,i])

        alpha[0,i] = np.arctan((y[1,i] - y[0,i])/(x[1,i] - x[0,i]))
        alpha[-1,i] = np.arctan((y[-1,i] - y[-2,i])/(x[-1,i] - x[-2,i]))
        alpha[1:-1,i] = (np.arctan( (y[1:-1,i] - y[0:-2,i])/(x[1:-1,i] - x[0:-2,i]) ) + \
                       np.arctan( (y[2:,i] - y[1:-1,i])/(x[2:,i] - x[1:-1,i]) )) /2.
        
        t = t + (f_year/1e3)*(1.881) #factor of 1.881 to convert from Mars-years to Earth-years
        t_obliq = np.floor(t)
        
        print ('t:', t)
        print ('t_obliq:', t_obliq)
        #print ('x[:,i]:', x[:,i])
        #print ('y[:,i]:', y[:,i])
        #print ('alpha[:,i]:', alpha[:,i]*180./np.pi)
        #print ('m_H2O[:,i-1]/920.:', m_H2O[:,i-1]/920.)
        #print ('m_CO2_max[:,i-1]:', m_CO2_max[:,i-1])
        #print ('T_max[:,i-1]:', T_max[:,i-1])
        
        #Save surface information
        if (N_runs == 0):
            np.save(path+'/x.npy', x[:,:])
            np.save(path+'/y.npy', y[:,:])
            np.save(path+'/alpha.npy', alpha[:,:])
            np.save(path+'/m_H2O.npy', m_H2O[:,:])
            np.save(path+'/m_CO2_max.npy', m_CO2_max[:,:])
            np.save(path+'/T_max.npy', T_max[:,:])

            #np.save(path+'/T_'+str(i-1)+'.npy', mars.T[:,0,:])
            #np.save(path+'/m_H2Ot_'+str(i-1)+'.npy', mars.m_H2Ot[:,:])
            #np.save(path+'/m_mol_'+str(i-1)+'.npy', mars.m_mol)
            #np.save(path+'/m_free_'+str(i-1)+'.npy', mars.m_H2O_free)
            #np.save(path+'/rho_vap_'+str(i-1)+'.npy', mars.rho_vap)
            #np.save(path+'/rho_atm_'+str(i-1)+'.npy', mars.rho_atm)
            #np.save(path+'/rho_sat_'+str(i-1)+'.npy', mars.rho_sat)
            #np.save(path+'/Qst_'+str(i-1)+'.npy', mars.Qst)
            #np.save(path+'/Q_solar_'+str(i-1)+'.npy', mars.Q_solart)
            #np.save(path+'/Q_IR_'+str(i-1)+'.npy', mars.Q_IRt)
            #np.save(path+'/Q_scat_'+str(i-1)+'.npy', mars.Q_scatt)
            #np.save(path+'/Q_surf_'+str(i-1)+'.npy', mars.Q_surft)
            
            #nu_mars = mars.nut*180./np.pi
            #Ls_mars = nu_mars + (mars.planet.Lp*180./np.pi)
            #np.save(path+'/Ls_mars_'+str(i-1)+'.npy', Ls_mars)
            #np.save(path+'/Ls_'+str(i-1)+'.npy', mars.Lst)
        
            #Save atmospheric information
            #np.save(path+'/atm_'+str(i-1)+'_'+path+'.npy', mars.atm_t)
            #np.save(path+'/atm1_'+str(i-1)+'_'+path+'.npy', mars.atm1_t)
            #np.save(path+'/atm_ice_flux_'+str(i-1)+'_'+path+'.npy', mars.atm_ice_flux_t)
        else:
            np.save(run_path+subpath+'/x.npy', x[:,:])
            np.save(run_path+subpath+'/y.npy', y[:,:])
            np.save(run_path+subpath+'/alpha.npy', alpha[:,:])
            np.save(run_path+subpath+'/m_H2O.npy', m_H2O[:,:])
            np.save(run_path+subpath+'/m_CO2_max.npy', m_CO2_max[:,:])
            np.save(run_path+subpath+'/T_max.npy', T_max[:,:])

            #np.save(run_path+subpath+'/T_'+str(i-1)+'.npy', mars.T[:,0,:])
            #np.save(run_path+subpath+'/m_H2Ot_'+str(i-1)+'.npy', mars.m_H2Ot[:,:])
            #np.save(run_path+subpath+'/rho_vap_'+str(i-1)+'.npy', mars.rho_vap)
            #np.save(run_path+subpath+'/rho_atm_'+str(i-1)+'.npy', mars.rho_atm)
            #np.save(run_path+subpath+'/rho_sat_'+str(i-1)+'.npy', mars.rho_sat)
            #nu_mars = mars.nut*180./np.pi
            #Ls_mars = nu_mars + (mars.planet.Lp*180./np.pi)
            #np.save(run_path+subpath+'/Ls_mars_'+str(i-1)+'.npy', Ls_mars)
        
        now = time.time()
        print ('Elapsed Time:', now-start)
        
    end = time.time()
    print ('Total Elapsed Time:', end-start)
        
    return x, y, path

def init_and_run(N_runs, run_path, run_num, dx, scale, length, n, f_year, t_start, **kwargs):
    #Initialize surface
    x0, y0, alpha0 = init_Surface(dx, scale, length)
    #Run surface evolution model
    
    if (f_year == 'var'):
        x, y, path = runSurfaceEvolution_fvar(n, x0, y0, alpha0, scale, N_runs=N_runs, run_path=run_path, run_num=run_num,\
                                              dx=dx, t_start=t_start, **kwargs)
    else:
        x, y, path = runSurfaceEvolution(n, x0, y0, alpha0, scale, N_runs=N_runs, run_path=run_path, run_num=run_num, dx=dx,\
                                         f_year=f_year, t_start=t_start, **kwargs)

def SimulRun(N_runs, dx, scale, length, n, f_year, t_start, **kwargs):
    #start = time.time()
    
    run_path = 'atmo5_2D_d'+str(length)+'m_R'+str(dx)+'m_VE'+str(scale)+'m_n'+str(n)+'_f'+str(f_year)+'yr_tstart'\
                               +str(t_start)+'_runs'+str(N_runs)
    
    path_num = 0
    while (os.path.exists(run_path) == True):
        run_path = 'atmo5_2D_d'+str(length+dx)+'m_R'+str(dx)+'m_VE'+str(scale)+'m_n'+str(n)+'_f'+str(f_year)\
                                   +'yr_tstart'+str(t_start)+'_runs'+str(N_runs)+'_'+str(path_num)
        path_num += 1
    os.mkdir(run_path)
    
    pool = mp.Pool(processes=N_runs)
    
    results = [pool.apply_async(init_and_run, args=(N_runs, run_path, i, dx, scale, length, n, f_year, t_start),\
                                     kwds=kwargs) for i in range(0, N_runs)]
    pool.close()
    
    #end = time.time()
    
    #print ('\n Total Time for Simultaneous Run:', end-start)

def runSurfaceEvolution_fvar(n, x0, y0, alpha0, scale, N_runs=0, run_path='', run_num=0, dx=0.1, planet=planets.Mars,\
                        lat=(85.*(np.pi/180)), ndays=668.59692, Ls=0, beta=0, f_IR=0.04, f_scat=0.02, u_wind=3., b_exp=0.2,\
                        atmExt=True, elev=-3., t_start=-1, dt=0, A_ice=0.37, shadow=True, yaxis_ext=True, por=0.5, \
                        z_rho=10., density='linear', N_atm=16, N_skin=7, Ls_eq=0, f_min=1., f_max=100.,\
                        vap_set=1, vap_set1=1, vap_Ls_start=0, vap_Ls_end=0, save_annual=False):
    # Interpolated Version Surface Evolution Function
    #n: Number of surfaces in evolved surface array. The number of evolution models run will be n-1.
    #x0, y0, alpha: length, height, and slope arrays of initial surface
    #scale: standard deviation of gaussian used to initialize vertial surface roughness
    #dx: horizontal resolution of points making up surface
    #f_year: Evolution timestep. Number of years (or fraction of year) over which mass balance is estimated to be unchanged.
    #######  Variable f_year in this function.
    start = time.time()
    
    rho_ice = 920.
    
    f_year = np.zeros([n-1])
    
    x = np.zeros([x0.size, n])
    y = np.zeros([y0.size, n])
    alpha = np.zeros([alpha0.size, n])

    m_H2O = np.zeros([x0.size, n-1])
    m_CO2_max = np.zeros([x0.size, n-1])
    T_max = np.zeros([x0.size, n-1])

    x[:,0] = x0
    y[:,0] = y0
    alpha[:,0] = alpha0
    
    #Make new directory or update existing directory
    if (N_runs == 0):
        path = 'atmo5_2D_d'+str(np.amax(x[:,0]))+'m_R'+str(dx)+'m_VE'+str(scale)+'m_n'+str(n)+'_fvar_tstart'+str(t_start)+\
               '_vap'+str(vap_set)
        path_num = 0
        while (os.path.exists(path) == True):
            path = 'atmo5_2D_d'+str(np.amax(x[:,0]))+'m_R'+str(dx)+'m_VE'+str(scale)+'m_n'+str(n)+'_fvar_tstart'+str(t_start)+\
                   '_vap'+str(vap_set)+'_'+str(path_num)
            path_num += 1
        os.mkdir(path)
    else:
        subpath = '/run_' + str(run_num)
        if os.path.exists(run_path+subpath):
            shutil.rmtree(run_path+subpath)
        os.mkdir(run_path+subpath)
    
    t = t_start #Time in kyr from present. Ex: t=-1 == 1000yr before present.
    t_obliq = t
    
    for i in range(1, n):
        print ('\ni:', i)
        x1 = np.array([x[:,i-1],y[:,i-1]])
        alpha1 = alpha[:,i-1]
        
        mars = model(planet=planet, lat=lat, ndays=ndays, Ls=Ls, x=x1, alpha=alpha1, beta=beta, \
                                         f_IR=f_IR, f_scat=f_scat, u_wind=u_wind, b_exp=b_exp, \
                                         atmExt=atmExt, elev=elev, t_obliq=t_obliq, dt=dt, A_ice=A_ice, dx=dx, \
                                         shadow=shadow, yaxis_ext=yaxis_ext, por=por, z_rho=z_rho, density=density,\
                                         N_atm=N_atm, N_skin=N_skin, Ls_eq=Ls_eq, vap_set=vap_set,\
                                         vap_set1=vap_set1, vap_Ls_start=vap_Ls_start, vap_Ls_end=vap_Ls_end)
        mars.run()
        
        m_H2O[:,i-1] = mars.m_H2Ot[-1,:]
        m_CO2_max[:,i-1] = np.amax(mars.m_CO2t[:,:], axis=0)
        T_max[:,i-1] = np.amax(mars.T[:,0,:], axis=0)
        
        z = np.amax( np.abs( (m_H2O[:-1,i-1]-m_H2O[1:,i-1])/(mars.profile.rho_ave) ) )
        f0 = dx/z
        if (f0 >= f_max):
            f_year[i-1] = f_max
        else:
            mag = np.floor(np.log10(f0))
            f_year[i-1] = 10**mag
        
        if (f_year[i-1] < f_min):
            f_year[i-1] = f_min
        
        #if (f_year[i-1] > f_max):
        #    f_year[i-1] = f_max
        
        print ('f_year:', f_year[i-1])
        
        x[:,i] = x[:,i-1] + f_year[i-1]*(m_H2O[:,i-1]/(mars.profile.rho_ave))*(-np.sin(alpha1*np.pi/180.))
        y[:,i] = y[:,i-1] + f_year[i-1]*(m_H2O[:,i-1]/(mars.profile.rho_ave))*(np.cos(alpha1*np.pi/180.))

        f = interp1d(x[:,i], y[:,i], kind='cubic', fill_value='extrapolate')

        x[:,i] = x0
        y[:,i] = f(x[:,i])

        alpha[0,i] = np.arctan((y[1,i] - y[0,i])/(x[1,i] - x[0,i]))
        alpha[-1,i] = np.arctan((y[-1,i] - y[-2,i])/(x[-1,i] - x[-2,i]))
        alpha[1:-1,i] = (np.arctan( (y[1:-1,i] - y[0:-2,i])/(x[1:-1,i] - x[0:-2,i]) ) + \
                       np.arctan( (y[2:,i] - y[1:-1,i])/(x[2:,i] - x[1:-1,i]) )) /2.
        
        t = t + (f_year[i-1]/1e3)*(1.881) #factor of 1.881 to convert from Mars-years to Earth-years
        t_obliq = np.floor(t)
        
        print ('t:', t)
        print ('t_obliq:', t_obliq)
        #print ('x[:,i]:', x[:,i])
        #print ('y[:,i]:', y[:,i])
        #print ('alpha[:,i]:', alpha[:,i]*180./np.pi)
        #print ('m_H2O[:,i-1]/920.:', m_H2O[:,i-1]/920.)
        #print ('m_CO2_max[:,i-1]:', m_CO2_max[:,i-1])
        #print ('T_max[:,i-1]:', T_max[:,i-1])
        
        #Save surface information
        if (N_runs == 0):
            np.save(path+'/x.npy', x[:,:])
            np.save(path+'/y.npy', y[:,:])
            np.save(path+'/alpha.npy', alpha[:,:])
            np.save(path+'/m_H2O.npy', m_H2O[:,:])
            np.save(path+'/m_CO2_max.npy', m_CO2_max[:,:])
            np.save(path+'/T_max.npy', T_max[:,:])
            np.save(path+'/f_year.npy', f_year)
            
            if (save_annual == True):
                np.save(path+'/T_'+str(i-1)+'.npy', mars.T[:,0,:])
                np.save(path+'/m_H2Ot_'+str(i-1)+'.npy', mars.m_H2Ot[:,:])
                np.save(path+'/m_CO2t_'+str(i-1)+'.npy', mars.m_CO2t[:,:])
                np.save(path+'/rho_vap_'+str(i-1)+'.npy', mars.rho_vap)
                np.save(path+'/rho_atm_'+str(i-1)+'.npy', mars.rho_atm)
                np.save(path+'/rho_sat_'+str(i-1)+'.npy', mars.rho_sat)
                nu_mars = mars.nut*180./np.pi
                Ls_mars = nu_mars + (mars.planet.Lp*180./np.pi)
                np.save(path+'/Ls_mars_'+str(i-1)+'.npy', Ls_mars)
        
            #Save atmospheric information
            #np.save(path+'/atm_'+str(i-1)+'_'+path+'.npy', mars.atm_t)
            #np.save(path+'/atm1_'+str(i-1)+'_'+path+'.npy', mars.atm1_t)
            #np.save(path+'/atm_ice_flux_'+str(i-1)+'_'+path+'.npy', mars.atm_ice_flux_t)
        else:
            np.save(run_path+subpath+'/x.npy', x[:,:])
            np.save(run_path+subpath+'/y.npy', y[:,:])
            np.save(run_path+subpath+'/alpha.npy', alpha[:,:])
            np.save(run_path+subpath+'/m_H2O.npy', m_H2O[:,:])
            np.save(run_path+subpath+'/m_CO2_max.npy', m_CO2_max[:,:])
            np.save(run_path+subpath+'/T_max.npy', T_max[:,:])
            np.save(run_path+subpath+'/f_year.npy', f_year)
            
            if (save_annual == True):
                np.save(run_path+subpath+'/T_'+str(i-1)+'.npy', mars.T[:,0,:])
                np.save(run_path+subpath+'/m_H2Ot_'+str(i-1)+'.npy', mars.m_H2Ot[:,:])
                np.save(run_path+subpath+'/m_CO2t_'+str(i-1)+'.npy', mars.m_CO2t[:,:])
                np.save(run_path+subpath+'/rho_vap_'+str(i-1)+'.npy', mars.rho_vap)
                np.save(run_path+subpath+'/rho_atm_'+str(i-1)+'.npy', mars.rho_atm)
                np.save(run_path+subpath+'/rho_sat_'+str(i-1)+'.npy', mars.rho_sat)
                nu_mars = mars.nut*180./np.pi
                Ls_mars = nu_mars + (mars.planet.Lp*180./np.pi)
                np.save(run_path+subpath+'/Ls_mars_'+str(i-1)+'.npy', Ls_mars)
        
        now = time.time()
        print ('Elapsed Time:', now-start)
        
    end = time.time()
    print ('Total Elapsed Time:', end-start)
        
    return x, y, path

def runSurfaceEvolutionContinue_fvar(cont, n, scale, t_start_cont=-1, N_runs=0, run_path='', run_num=0, dx=0.1,\
                        planet=planets.Mars, lat=(85.*(np.pi/180)), ndays=668.59692, Ls=0, beta=0, f_IR=0.04, f_scat=0.02,\
                        u_wind=3., b_exp=0.2, atmExt=True, elev=-3., dt=0, A_ice=0.37, shadow=True, yaxis_ext=True,\
                        por=0.5, z_rho=10., density='linear', N_atm=16, N_skin=7, Ls_eq=0, f_min=1., f_max=100.,\
                        vap_set=1, vap_set1=1, vap_Ls_start=0, vap_Ls_end=0, save_annual=False):
    #This version continues the surface evolution model from where it left off in a previous model run
    # Interpolated Version Surface Evolution Function
    #cont: path to previous model run
    #n: Number of surfaces in evolved surface array. The number of evolution models run will be n-1.
    #scale: standard deviation of gaussian used to initialize vertial surface roughness
    #dx: horizontal resolution of points making up surface
    #f_year: Evolution timestep. Number of years (or fraction of year) over which mass balance is estimated to be unchanged.
    #######  Variable f_year in this function.
    start = time.time()
    
    xCont = np.load(cont+'/x.npy')
    yCont = np.load(cont+'/y.npy')
    alphaCont = np.load(cont+'/alpha.npy')
    m_H2OCont = np.load(cont+'/m_H2O.npy')
    m_CO2_maxCont = np.load(cont+'/m_CO2_max.npy')
    T_maxCont = np.load(cont+'/T_max.npy')
    f_yearCont = np.load(cont+'/f_year.npy')
    
    nCont = xCont.shape[1]
    
    rho_ice = 920.
    
    x = np.append( xCont, np.zeros([xCont.shape[0], n]), axis=1 )
    y = np.append( yCont, np.zeros([yCont.shape[0], n]), axis=1 )
    alpha = np.append( alphaCont, np.zeros([alphaCont.shape[0], n]), axis=1 )

    m_H2O = np.append( m_H2OCont, np.zeros([m_H2OCont.shape[0], n]), axis=1 )
    m_CO2_max = np.append( m_CO2_maxCont, np.zeros([m_CO2_maxCont.shape[0], n]), axis=1 )
    T_max = np.append( T_maxCont, np.zeros([T_maxCont.shape[0], n]), axis=1 )
    f_year = np.append( f_yearCont, np.zeros(n) )
    
    #Make new directory or update existing directory
    if (N_runs == 0):
        path = 'atmo5_2D_d'+str(np.amax(x[:,0]))+'m_R'+str(dx)+'m_VE'+str(scale)+'m_n'+str(n+nCont)+'_fvar_tstart'+\
               str(t_start_cont)+'_vap'+str(vap_set)
        path_num = 0
        while (os.path.exists(path) == True):
            path = 'atmo5_2D_d'+str(np.amax(x[:,0]))+'m_R'+str(dx)+'m_VE'+str(scale)+'m_n'+str(n+nCont)+'_fvar_tstart'+\
               str(t_start_cont)+'_vap'+str(vap_set)+'_'+str(path_num)
            path_num += 1
        os.mkdir(path)
    else:
        subpath = '/run_' + str(run_num)
        if os.path.exists(run_path+subpath):
            shutil.rmtree(run_path+subpath)
        os.mkdir(run_path+subpath)
    
    t = t_start_cont + (np.sum(f_yearCont)/1e3)*(1.881) #Time in kyr from present. Ex: t=-1 == 1000yr before present.
    t_obliq = np.floor(t)
    
    for i in range(nCont, n+nCont):
        print ('\ni:', i)
        x1 = np.array([x[:,i-1],y[:,i-1]])
        alpha1 = alpha[:,i-1]
        
        mars = model(planet=planet, lat=lat, ndays=ndays, Ls=Ls, x=x1, alpha=alpha1, beta=beta, \
                                         f_IR=f_IR, f_scat=f_scat, u_wind=u_wind, b_exp=b_exp, \
                                         atmExt=atmExt, elev=elev, t_obliq=t_obliq, dt=dt, A_ice=A_ice, dx=dx, \
                                         shadow=shadow, yaxis_ext=yaxis_ext, por=por, z_rho=z_rho, density=density,\
                                         N_atm=N_atm, N_skin=N_skin, Ls_eq=Ls_eq, vap_set=vap_set,\
                                         vap_set1=vap_set1, vap_Ls_start=vap_Ls_start, vap_Ls_end=vap_Ls_end)
        mars.run()
        
        m_H2O[:,i-1] = mars.m_H2Ot[-1,:]
        m_CO2_max[:,i-1] = np.amax(mars.m_CO2t[:,:], axis=0)
        T_max[:,i-1] = np.amax(mars.T[:,0,:], axis=0)
        
        z = np.amax( np.abs( (m_H2O[:-1,i-1]-m_H2O[1:,i-1])/(mars.profile.rho_ave) ) )
        f0 = dx/z
        if (f0 >= f_max):
            f_year[i-1] = f_max
        else:
            mag = np.floor(np.log10(f0))
            f_year[i-1] = 10**mag
        
        if (f_year[i-1] < f_min):
            f_year[i-1] = f_min
        
        #if (f_year[i-1] > f_max):
        #    f_year[i-1] = f_max
        
        print ('f_year:', f_year[i-1])
        
        x[:,i] = x[:,i-1] + f_year[i-1]*(m_H2O[:,i-1]/(mars.profile.rho_ave))*(-np.sin(alpha1*np.pi/180.))
        y[:,i] = y[:,i-1] + f_year[i-1]*(m_H2O[:,i-1]/(mars.profile.rho_ave))*(np.cos(alpha1*np.pi/180.))

        f = interp1d(x[:,i], y[:,i], kind='cubic', fill_value='extrapolate')

        x[:,i] = x[:,0]
        y[:,i] = f(x[:,i])

        alpha[0,i] = np.arctan((y[1,i] - y[0,i])/(x[1,i] - x[0,i]))
        alpha[-1,i] = np.arctan((y[-1,i] - y[-2,i])/(x[-1,i] - x[-2,i]))
        alpha[1:-1,i] = (np.arctan( (y[1:-1,i] - y[0:-2,i])/(x[1:-1,i] - x[0:-2,i]) ) + \
                       np.arctan( (y[2:,i] - y[1:-1,i])/(x[2:,i] - x[1:-1,i]) )) /2.
        
        t = t + (f_year[i-1]/1e3)*(1.881) #factor of 1.881 to convert from Mars-years to Earth-years
        t_obliq = np.floor(t)
        
        print ('t:', t)
        print ('t_obliq:', t_obliq)
        #print ('x[:,i]:', x[:,i])
        #print ('y[:,i]:', y[:,i])
        #print ('alpha[:,i]:', alpha[:,i]*180./np.pi)
        #print ('m_H2O[:,i-1]/920.:', m_H2O[:,i-1]/920.)
        #print ('m_CO2_max[:,i-1]:', m_CO2_max[:,i-1])
        #print ('T_max[:,i-1]:', T_max[:,i-1])
        
        #Save surface information
        if (N_runs == 0):
            np.save(path+'/x.npy', x[:,:])
            np.save(path+'/y.npy', y[:,:])
            np.save(path+'/alpha.npy', alpha[:,:])
            np.save(path+'/m_H2O.npy', m_H2O[:,:])
            np.save(path+'/m_CO2_max.npy', m_CO2_max[:,:])
            np.save(path+'/T_max.npy', T_max[:,:])
            np.save(path+'/f_year.npy', f_year)
            
            if (save_annual == True):
                np.save(path+'/T_'+str(i-1)+'.npy', mars.T[:,0,:])
                np.save(path+'/m_H2Ot_'+str(i-1)+'.npy', mars.m_H2Ot[:,:])
                np.save(path+'/m_CO2t_'+str(i-1)+'.npy', mars.m_CO2t[:,:])
                np.save(path+'/rho_vap_'+str(i-1)+'.npy', mars.rho_vap)
                np.save(path+'/rho_atm_'+str(i-1)+'.npy', mars.rho_atm)
                np.save(path+'/rho_sat_'+str(i-1)+'.npy', mars.rho_sat)
                nu_mars = mars.nut*180./np.pi
                Ls_mars = nu_mars + (mars.planet.Lp*180./np.pi)
                np.save(path+'/Ls_mars_'+str(i-1)+'.npy', Ls_mars)
        
            #Save atmospheric information
            #np.save(path+'/atm_'+str(i-1)+'_'+path+'.npy', mars.atm_t)
            #np.save(path+'/atm1_'+str(i-1)+'_'+path+'.npy', mars.atm1_t)
            #np.save(path+'/atm_ice_flux_'+str(i-1)+'_'+path+'.npy', mars.atm_ice_flux_t)
        else:
            np.save(run_path+subpath+'/x.npy', x[:,:])
            np.save(run_path+subpath+'/y.npy', y[:,:])
            np.save(run_path+subpath+'/alpha.npy', alpha[:,:])
            np.save(run_path+subpath+'/m_H2O.npy', m_H2O[:,:])
            np.save(run_path+subpath+'/m_CO2_max.npy', m_CO2_max[:,:])
            np.save(run_path+subpath+'/T_max.npy', T_max[:,:])
            np.save(run_path+subpath+'/f_year.npy', f_year)
            
            if (save_annual == True):
                np.save(run_path+subpath+'/T_'+str(i-1)+'.npy', mars.T[:,0,:])
                np.save(run_path+subpath+'/m_H2Ot_'+str(i-1)+'.npy', mars.m_H2Ot[:,:])
                np.save(run_path+subpath+'/m_CO2t_'+str(i-1)+'.npy', mars.m_CO2t[:,:])
                np.save(run_path+subpath+'/rho_vap_'+str(i-1)+'.npy', mars.rho_vap)
                np.save(run_path+subpath+'/rho_atm_'+str(i-1)+'.npy', mars.rho_atm)
                np.save(run_path+subpath+'/rho_sat_'+str(i-1)+'.npy', mars.rho_sat)
                nu_mars = mars.nut*180./np.pi
                Ls_mars = nu_mars + (mars.planet.Lp*180./np.pi)
                np.save(run_path+subpath+'/Ls_mars_'+str(i-1)+'.npy', Ls_mars)
        
        now = time.time()
        print ('Elapsed Time:', now-start)
        
    end = time.time()
    print ('Total Elapsed Time:', end-start)
        
    return x, y, path