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
#mnn = [1915.30785163,62.916694645,785894328.86,13708.1158646,516.469237568,-28219159894.4,6070.40446623,182.739289213,27783119636.5,4109.87529296,558.211162464,59909192419.6,-3797.28033274,4406.68073528,34817507.7047,689.096226417,495.512764554,1935731384.93]

# CMA-ES 6 discs, Chi-sq = 1.44e5, nite = 26601
mnn = [4.30447325e+03,   3.47631850e+02,  -1.50800635e+05,  -3.82377879e+03,
       4.39650257e+03,   3.27170501e-02,   9.03293364e+02,   3.62306261e+02,
       2.56145113e+00,   4.28953554e+03,   3.51922961e+02,   8.28977907e+04,
       1.53210805e+04,   5.35579774e+02,  -2.37197310e+01,   4.32295662e+03,
       3.42512426e+02,   6.79858270e+04]


# CMA-ES 7 discs, Chi-sq = 2.89e4, nite = 10000
mnn = [5.42788657e+03,   7.41958749e+02,   6.66772454e+01,   4.89817833e+03,
       1.94342650e+02,   3.63781597e+01,  -5.88843555e+05,   6.01317933e+05,
       -3.68081316e+05,  -5.53295016e+03,   6.34532356e+03,   2.56063810e-02,
       9.55189517e+02,   5.10455991e+02,   4.70079616e+00,  -5.88931927e+05,
       6.01406318e+05,   3.68080218e+05,   1.10932778e+04,   6.78636222e+02,
       -4.36247904e+01]

# CMA-ES 8 discs, Chi-sq = 3.31e4, nite = 50000
mnn = [2.15890657e+03 ,  3.45335791e+03 , -2.22826199e+00 ,  1.06865603e+03,
       7.55340857e+02 ,  3.00489325e+01 ,  1.14539435e+04 ,  6.82893394e+02,
       -3.46930410e+01,   5.01652573e+03,   2.13040329e+02,   4.11216091e+01,
       4.70954175e+02 ,  5.16308818e+01 ,  3.93953467e-02 , -1.56330091e+03,
       3.14414096e+03 ,  4.79815196e-01 ,  1.05452880e+03 ,  8.50824727e+02,
       -2.65445284e+01,   4.75849553e+03,   8.25111359e+02,   5.58979864e+01]

# CMA-ES 9 discs, Chi-sq = 1.75e4, nite = 50000
mnn = [3.92316056e+02,   5.57181942e+01,   2.99329828e-02,  -3.69101888e+03,
       1.57509946e+04,   7.00862706e+03,   1.22792161e+03,   7.45830408e+02,
       1.26482174e+03,   5.02167633e+03,   2.02605336e+02,   3.77916003e+01,
       1.03150142e+04,   7.12990100e+02,  -5.43567495e+01,   5.83739110e+03,
       7.80292453e+02,   7.43470795e+01,   1.22722960e+03,   7.47697550e+02,
       -1.25780637e+03,  -3.68987711e+03,   1.57491447e+04,  -7.00980364e+03,
       -3.22803071e+03,   4.24900168e+03,   9.31700146e-02]

# CMA-ES 10 discs, Chi-sq = 2.35e4, nite = 50000
mnn = [1.64058020e+03,   5.11149191e+02,   9.00339378e+00,   7.52320109e+03,
       6.52015901e+02,  -3.12115447e+02,  -3.62872724e+03,   4.59457408e+03,
       6.50962596e-02,   6.83068040e+03,   6.82348502e+02,   3.31138069e+02,
       3.94565136e+01,   5.10012036e+02,   1.47844337e-01,   2.88069514e+03,
       5.20797517e+01,   1.29740724e+00,   5.65442344e+03 ,  2.33587583e+02,
       4.27680883e+01,   9.58909564e+03,   1.79502700e+03 , -5.16057165e+02,
       7.54131059e+03,   2.27543152e+03,  -4.61250497e+01,   9.38976404e+03,
       1.83899397e+03,   5.54069256e+02]

# CMA-ES 10 discs, new
mnn = [1.07774710e+04,   9.62098608e+02,  -2.37363610e+03,  -1.97219971e+03,
       3.09136156e+03,   1.42508524e-01,   7.47815347e+03,   2.54267803e+02,
       5.16123920e+01,   1.11319807e+04,   3.99727661e+02,  -4.41718431e+01,
       1.26260109e+04,   9.28133948e+02,  -3.05294506e+03,   2.73723749e+03,
       4.54369342e+02,   3.85772084e+01,   1.09577182e+04,   9.60744728e+02,
       3.39203831e+03,   1.31804811e+04,   9.13639886e+02,   2.05453791e+03,
       4.75733657e+03,   8.08716353e+03,  -6.16700640e+00,   6.37968513e+03,
       8.99925571e+03,   4.72246532e+00]

mnn = [6.07975702e+03,   1.29669366e+03,  -5.19550852e+03,   5.15932991e+03,
       8.55804530e+03,  -4.98147474e+02,   2.86647223e+03,   4.99350374e+02,
       4.13125442e+01,   7.43748636e+03,   2.88698678e+02,   4.32647741e+02,
       5.17651025e+03,   8.56868978e+03,   4.96610561e+02,   6.41101882e+03,
       1.28557637e+03,   9.13879477e+04,   6.43196040e+03,   1.28491942e+03,
       -8.61852954e+04,   6.05657274e+02,   2.05517115e+03,   2.12404470e+00,
       7.71832696e+03,   3.02048391e+02,  -4.16658084e+02]

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





