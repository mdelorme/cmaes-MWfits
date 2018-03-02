#import matplotlib as mpl
#mpl.use('pdf')
import numpy as np
import matplotlib.pyplot as plt
from mnn.model import MNnModel
from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter

plot_fit_points = True
fit_data = 'data/rho_disc_restricted.dat'
fit = np.loadtxt(fit_data)

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

print('Plotting discs : ')
print(' - Generating log spaces')
Np   = 1024
TotP = Np**2.0
R0 = 25000.0  # pc
Z0 = 5000.0   # pc 
R  = np.logspace(-3.0, 0.0, Np) * R0
Z  = np.logspace(-3.0, 0.0, Np) * Z0 
Rmin = R.min()

print(' - Generating grid')

def disc_density(R, z):
    rho_thin  = stellar_disc_density(R, z, sig0_thin,  zd_thin,  Rd_thin)
    rho_thick = stellar_disc_density(R, z, sig0_thick, zd_thick, Rd_thick)
    rho_HI    = gas_disc_density(R, z, sig0_HI, zd_HI, Rm_HI, Rd_HI)
    rho_H2    = gas_disc_density(R, z, sig0_H2, zd_H2, Rm_H2, Rd_H2)

    return rho_thin + rho_thick + rho_HI + rho_H2

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111)
zmin = Z.min()
Rmin = R.min()

print(' - Computing disc density')
rho_R = disc_density(R, zmin)
rho_z = disc_density(Rmin, Z)
    
print(' - Building MNn model')
mnn = [-1.04675003e+05,   1.05832671e+05,   1.51673538e-03,   1.99290602e+03,
       1.80848385e+03,  -4.46458988e+02,   1.72733282e+03,   1.70579443e+03,
       2.15215380e+02,   2.64931787e+03,   2.23904931e+03,  -8.38751344e+02,
       2.84629186e+03,   2.12242469e+03,   3.78193600e+02,   5.79224620e+03,
       7.81988040e+01,   2.85482938e+01,   2.43668197e+03,   2.18347057e+03,
       6.94240808e+02,   3.44944272e+03,   5.47265099e+02,   5.95155501e+01,
       3.27528042e+04,   5.93579966e+02,  -4.54720125e+01]

ndisc = len(mnn) // 3
for i in range(ndisc):
    mnn[i*3+2] *= 1.0e9

    
model = MNnModel()
for i in range(ndisc):
    model.add_disc('z', mnn[i*3], mnn[i*3+1], mnn[i*3+2])

s = ' - Checking definite-positiveness at 1 Mpc: '
if not model.is_positive_definite(1.0e6):
    s += 'NOT DP !'
else:
    s += 'OK'
print(s)

print(' - Computing MNn densities')

dat = np.loadtxt('data/rho_disc.dat')
Rmask = (dat[:,2] == dat[:,2].min())
Zmask = (dat[:,0] == dat[:,0].min())

Np = 50
R_fit  = np.logspace(-3.0, 0.0, Np) * R0
Z_fit  = np.logspace(-3.0, 0.0, Np) * Z0

rho_mnn_R = model.evaluate_density(R, 0.0, zmin)
rho_mnn_Z = model.evaluate_density(Rmin, 0.0, Z)

rho_fit_R = model.evaluate_density(R_fit, 0.0, zmin)
rho_fit_Z = model.evaluate_density(Rmin, 0.0, Z_fit)
rho_obs_R = disc_density(R_fit, zmin)
rho_obs_Z = disc_density(Rmin, Z_fit)

# Rand z for fit
print(' - Computing residuals')
res_R = (rho_mnn_R - rho_R) / rho_R
res_Z = (rho_mnn_Z - rho_z) / rho_z

res_fit_R = (rho_fit_R - rho_obs_R) / rho_obs_R
res_fit_Z = (rho_fit_Z - rho_obs_Z) / rho_obs_Z



R *= 1.0e-3
R_fit *= 1.0e-3
Z *= 1.0e-3
Z_fit *= 1.0e-3

ax1 = plt.subplot2grid((4, 2), (0, 0), rowspan=3)
ax1.plot(R, np.log10(rho_mnn_R), '-b', label='MNn')
ax1.plot(R_fit, np.log10(rho_fit_R), '+b')
ax1.plot(R, np.log10(rho_R), '--k',  label='McMillan')
ax1.set_ylabel(r'$log(\rho) [M_\odot.pc^{-3}]$')
#ax1.set_xlim((0.0, 25.0))
ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax2 = plt.subplot2grid((4, 2), (0, 1), rowspan=3, sharey=ax1)
ax2.plot(Z, np.log10(rho_mnn_Z), '-b', label='MNn')
ax2.plot(Z_fit, np.log10(rho_fit_Z), '+b')
ax2.plot(Z, np.log10(rho_z),  '--k', label='McMillan')
ax2.set_ylabel(r'$log(\rho) [M_\odot.pc^{-3}]$')
#ax2.set_xlim((0.0, 25.0))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax3 = plt.subplot2grid((4, 2), (3, 0), sharex=ax1)
ax3.plot((0.0, R.max()), (0.0, 0.0), '--k')
ax3.plot(R, res_R, 'red')
ax3.plot(R_fit, res_fit_R, '+r')
ax3.set_xlabel(r'$R [kpc]$')
ax3.set_ylabel(r'$\frac{\rho_{mnn} - \rho}{\rho}$')
#ax3.set_xlim((0.0, 25.0))
ax3.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax4 = plt.subplot2grid((4, 2), (3, 1), sharey=ax3, sharex=ax2)
ax4.plot((0.0, Z.max()), (0.0, 0.0), '--k')
ax4.plot(Z, res_Z, 'red')
ax4.plot(Z_fit, res_fit_Z, '+r')
ax4.set_xlabel(r'$z [kpc]$')
ax4.set_ylabel(r'$\frac{\rho_{mnn} - \rho}{\rho}$')
#ax4.set_xlim((0.0, 25.0))
ax4.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax4.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.tight_layout()
plt.savefig('plots/discs_fit.pdf')
#plt.show()

# Diagonal
print(' - Building the (R, 0.2R) line')
rho_o = disc_density(R*1.0e3, 0.2*R*1.0e3)
rho_m = model.evaluate_density(R*1.0e3, 0.0, 0.2*R*1.0e3)
rho_R = (rho_o - rho_m) / rho_o

fig = plt.figure(figsize=(9, 10))
rd = np.sqrt(R**2.0 + (0.2*R)**2.0)
ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
ax1.plot(rd, np.log10(rho_o), '--k')
ax1.plot(rd, np.log10(rho_m), '-b')
ax1.set_xlabel(r'$||r|| [kpc]$')
ax1.set_ylabel(r'$log_{10}(\rho / (M_\odot . pc^{-3}))$')

ax2 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)
ax2.plot((0.0, rd.max()), (0.0, 0.0), '--k')
ax2.plot(rd, rho_R, '-r')
ax2.set_xlabel(r'$||r|| [kpc]$')
ax2.set_ylabel(r'$\frac{\rho_{mnn} - \rho}{\rho}$')
plt.tight_layout()
plt.savefig('plots/discs_fit_diag.pdf')

# In 2d
print(' - Building values in 2D')
Rg, Zg = np.meshgrid(R*1.0e3, Z*1.0e3)
rho_o = disc_density(Rg, Zg)
rho_m = model.evaluate_density(Rg, 0.0, Zg)

clim = (min(rho_o.min(), rho_m.min()), max(rho_o.max(), rho_m.max()))
print(clim)
extent = [R.min(), R.max(), Z.min(), Z.max()]
print(extent)
fig = plt.figure(figsize=(15, 7))
ax1 = plt.subplot(121)
im1 = ax1.imshow(rho_o, clim=clim, extent=extent, origin='lower', aspect=5.0)
ax1.set_xlabel(r'$R [kpc]$')
ax1.set_ylabel(r'$z [kpc]$')
ax1.set_title('McMillan model')
ax2 = plt.subplot(122, sharey=ax1)
im2 = ax2.imshow(rho_m, clim=clim, extent=extent, origin='lower', aspect=5.0)
ax2.set_xlabel(r'$R [kpc]$')
ax2.set_title('MNn model')
fig.colorbar(im1, ax=(ax1, ax2), label=r'$\rho [M_\odot . pc^{-3}]$')
plt.savefig('plots/discs_2d.pdf')
#plt.show()


fig = plt.figure(figsize=(8, 8))

Rrho = np.log10(np.abs(rho_m - rho_o) / rho_o)
aspect = R.max() / Z.max()
Rrho_min = Rrho.min()
Rrho_max = Rrho.max()
clim = (Rrho_min, Rrho_max)
im = plt.imshow(Rrho, extent=extent, aspect=aspect, clim=clim, origin='lower')
plt.scatter(fit[:,0]*1.0e-3, fit[:,2]*1.0e-3, color='white', marker=',', s=0.05)

ax.set_xlim(fit[:,0].min(), fit[:,0].max())
ax.set_ylim(fit[:,2].min(), fit[:,2].max())
#plt.set_xlabel(r'$R [kpc]$')
#plt.set_ylabel(r'$z [kpc]$')
fig.colorbar(im, label=r'$log_{10}(|\frac{\rho_{mnn} - \rho}{\rho}|)$', shrink=0.8)
plt.xlabel(r'$R [kpc]$')
plt.ylabel(r'$z [kpc]$')
#plt.tight_layout()
fig.savefig('plots/discs_residuals_2d.pdf')





