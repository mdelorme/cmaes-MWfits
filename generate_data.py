import matplotlib as mpl
mpl.use('pdf')
from mnn.fitter import MNnFitter
from mnn.model import MNnModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.optimize import minimize
import sys, os

print('Building McMillan model')

# Values taken from McMillan 2016

#### Bulge
alpha = 1.8 
r0    = 75.0    # pc
rcut  = 2100.0  # pc
q     = 0.5
Mb    = 8.9e9   # Msun, not used in the formula
rho0b = 9.93e1  # Msun.pc^-3

def bulge_density(R, z):
    rp = np.sqrt(R**2.0 + (z / q)**2.0)
    return rho0b * np.exp(-(rp / rcut)**2.0) / (1.0 + rp/r0)**alpha

#### Stellar discs
zd_thin    = 300.0  # pc
zd_thick   = 900.0  # pc
Rd_thin    = 2500.0 # pc
Rd_thick   = 3600.0 # pc
sig0_thin  = 896.0  # Msun/pc^2
sig0_thick = 183.0  # Msun/pc^2

def stellar_disc_density(R, z, sig0, zd, Rd):
    return sig0 * np.exp(-np.abs(z) / zd - R / Rd) / (2.0 * zd)

#### Gas discs
sig0_HI = 53.1    # Msun/pc^2
sig0_H2 = 2180.0  # Msun/pc^2
zd_HI   = 85.0    # pc
zd_H2   = 45.0    # pc
Rm_HI   = 4000.0  # pc
Rm_H2   = 12000.0 # pc
Rd_HI   = 7000.0  # pc
Rd_H2   = 1500.0  # pc

def gas_disc_density(R, z, sig0, zd, Rm, Rd):
    zq = 0.5 * z / zd
    q = sig0 / (4.0*zd)
    sech2 = (2.0 / (np.exp(zq) + np.exp(-zq)))**2.0
    return q * np.exp(-Rm / R - R / Rd) * sech2

print(' - Generating log spaces')
Np   = 50
TotP = Np**2.0
R0_d = 25000.0 # pc
Z0_d = 5000.0  # pc
R0_b = 2100.0  # pc
Z0_b = 2100.0  # pc 
R_b  = np.logspace(-2.0, 0.0, Np) * R0_b
Z_b  = np.logspace(-2.0, 0.0, Np) * Z0_b
R_d  = np.logspace(-3.0, 0.0, Np) * R0_d
Z_d  = np.logspace(-3.0, 0.0, Np) * Z0_d

print(' - Generating grid')
Rg, Zg = np.meshgrid(R_d, Z_d)
Rv     = Rg.ravel()
Y      = np.zeros((Rv.shape[0], ))
Zv     = Zg.ravel()

Rgb, Zgb = np.meshgrid(R_b, Z_b)
Rvb    = Rgb.ravel()
Zvb    = Zgb.ravel()

print(' - Computing bulge density')

rho_bulge = bulge_density(Rvb, Zvb)
tab = np.stack((Rvb, Y, Zvb, rho_bulge)).T
np.savetxt('data/rho_b.dat', tab)

print(' - Computing stellar thin disk density')
rho_thin = stellar_disc_density(Rv, Zv, sig0_thin, zd_thin, Rd_thin)
tab = np.stack((Rv, Y, Zv, rho_thin)).T
np.savetxt('data/rho_thin.dat', tab)

print(' - Computing stellar thick disk density')
rho_thick = stellar_disc_density(Rv, Zv, sig0_thick, zd_thick, Rd_thick)
tab = np.stack((Rv, Y, Zv, rho_thick)).T
np.savetxt('data/rho_thick.dat', tab)

print(' - Computing HI gas disc density')
rho_HI = gas_disc_density(Rv, Zv, sig0_HI, zd_HI, Rm_HI, Rd_HI)
tab = np.stack((Rv, Y, Zv, rho_HI)).T
np.savetxt('data/rho_HI.dat', tab)


print(' - Computing H2 gas disc density')
rho_H2 = gas_disc_density(Rv, Zv, sig0_H2, zd_H2, Rm_H2, Rd_H2)
tab = np.stack((Rv, Y, Zv, rho_H2)).T
np.savetxt('data/rho_H2.dat', tab)

print(' - Summing disc contributions and saving to data/rho_disc.dat')
rho_disc = rho_thin + rho_thick + rho_HI + rho_H2

tab = np.stack((Rv, Y, Zv, rho_disc)).T
np.savetxt('data/rho_disc.dat', tab)

# No need to plot anymore
exit(0)

print('\nPlotting densities')
rho = rho_disc + rho_bulge
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(121)

Rmask = (Zv == Zv.min())
Zmask = (Rv == Rv.min())
RvR = Rv[Rmask]*1.0e-3 #np.log10(Rv[Rmask])

ax.plot(RvR, np.log10(rho_bulge[Rmask]), linestyle='--', color='red',     label='Bulge')
ax.plot(RvR, np.log10(rho_thin[Rmask]),  linestyle='--', color='blue',    label='Thin disc')
ax.plot(RvR, np.log10(rho_thick[Rmask]), linestyle='--', color='magenta', label='Thick disc')
ax.plot(RvR, np.log10(rho_HI[Rmask]),    linestyle='--', color='green',   label='HI disc')
ax.plot(RvR, np.log10(rho_H2[Rmask]),    linestyle='--', color='purple' , label='H2 disc')
ax.plot(RvR, np.log10(rho[Rmask]),                       color='black',   label='Summed model')
ax.legend()
#ax.set_xlim(1.0, 5.0)
#ax.set_xlim(0.0, 10000.0)
ax.set_ylim(-5.0, 5.0)
ax.set_xlabel(r'$R [kpc]$')
ax.set_ylabel(r'$log(\rho)$')

ax = fig.add_subplot(122)
ZvZ = Zv[Zmask] * 1.0e-3 #np.log10(Zv[Zmask])
ax.plot(ZvZ, np.log10(rho_bulge[Zmask]), linestyle='--', color='red',     label='Bulge')
ax.plot(ZvZ, np.log10(rho_thin[Zmask]),  linestyle='--', color='blue',    label='Thin disc')
ax.plot(ZvZ, np.log10(rho_thick[Zmask]), linestyle='--', color='magenta', label='Thick disc')
#ax.plot(ZvZ, np.log10(rho_HI[Zmask]),    linestyle='--', color='green',   label='HI disc')   These disks won't appear at R = 0
#ax.plot(ZvZ, np.log10(rho_H2[Zmask]),    linestyle='--', color='purple' , label='H2 disc')
ax.plot(ZvZ, np.log10(rho[Zmask]),                       color='black',   label='Summed model')
ax.legend()

ax.set_ylim(-5.0, 5.0)
ax.set_xlim(0.0, 5.0)
ax.set_xlabel(r'$Z [kpc]$')
ax.set_ylabel(r'$log(\rho)$')

fig.suptitle('McMillan model')
fig.savefig('plots/mcmillan.pdf')





