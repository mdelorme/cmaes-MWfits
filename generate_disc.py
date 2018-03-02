import numpy as np

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


R1 = np.linspace(50.0, 5.0e2, 5)
z1 = np.linspace(50.0, 1.0e2, 5)

R2 = np.linspace(5.0e2, 30.0e3, 50)
z2 = np.linspace(1.0e2, 5.0e3, 50)

Rg1, zg1 = np.meshgrid(R1, z1)
Rg2, zg2 = np.meshgrid(R2, z2)

Rg1 = Rg1.ravel()
zg1 = zg1.ravel()
Rg2 = Rg2.ravel()
zg2 = zg2.ravel()
R = np.concatenate((Rg1, Rg2))
z = np.concatenate((zg1, zg2))
print('Number of points : {}'.format(R.shape[0]))

print('Generating datasets for discs : ')

print(' - Computing stellar thin disk density')
rho_thin1 = stellar_disc_density(Rg1, zg1, sig0_thin, zd_thin, Rd_thin)
rho_thin2 = stellar_disc_density(Rg2, zg2, sig0_thin, zd_thin, Rd_thin)
#rho_thin3 = stellar_disc_density(R3, z3, sig0_thin, zd_thin, Rd_thin)

print(' - Computing stellar thick disk density')
rho_thick1 = stellar_disc_density(Rg1, zg1, sig0_thick, zd_thick, Rd_thick)
rho_thick2 = stellar_disc_density(Rg2, zg2, sig0_thick, zd_thick, Rd_thick)
#rho_thick3 = stellar_disc_density(R3, z3, sig0_thick, zd_thick, Rd_thick)

print(' - Computing HI gas disc density')
rho_HI1 = gas_disc_density(Rg1, zg1, sig0_HI, zd_HI, Rm_HI, Rd_HI)
rho_HI2 = gas_disc_density(Rg2, zg2, sig0_HI, zd_HI, Rm_HI, Rd_HI)
#rho_HI3 = gas_disc_density(R3, z3, sig0_HI, zd_HI, Rm_HI, Rd_HI)

print(' - Computing H2 gas disc density')
rho_H21 = gas_disc_density(Rg1, zg1, sig0_H2, zd_H2, Rm_H2, Rd_H2)
rho_H22 = gas_disc_density(Rg2, zg2, sig0_H2, zd_H2, Rm_H2, Rd_H2)
#rho_H23 = gas_disc_density(R3, z3, sig0_H2, zd_H2, Rm_H2, Rd_H2)

print(' - Stacking values')
#rho_thin1 = rho_thin1.ravel()
#rho_thin2 = rho_thin2.ravel()
#rho_thin3 = rho_thin3.ravel()
rho_thin  = np.concatenate((rho_thin1, rho_thin2))
rho_thick = np.concatenate((rho_thick1, rho_thick2))
rho_HI    = np.concatenate((rho_HI1, rho_HI2))
rho_H2    = np.concatenate((rho_H21, rho_H22))

rho_discs = rho_thin + rho_thick + rho_HI + rho_H2
Np = R.shape[0]
tab = np.stack((R, np.zeros((Np,)), z, rho_discs)).T
np.savetxt('data/rho_disc_restricted.dat', tab)

