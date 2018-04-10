from mnn.model import MNnModel
import numpy as np
import sys
import cma # pycma is the optimizer package we are using for the fits now
import mnn

# Number of discs is passed in parameters (default = 10)
ndiscs = int(sys.argv[1]) if len(sys.argv) > 0 else 10

# The data we are fitting, here it's a file with 5 columns : R, z, phi, dphi/dR, dphi/dz
dat    = np.loadtxt('pjm_model_disc.dat')
dat[:,3] *= -1.0 # dPhi/dR = -FR
dat[:,4] *= -1.0 # dPhi/dz = -Fz

# We try to minimize the residuals on each component of the force.
# These terms indicate what error we agree to take on the fit.
# Usually the term will be of the form 1.0 / (coeff*dat[:,3]**2.0). Here coeff = 1.0
inv_sig2_R = 1.0 / dat[:,3]**2.0
inv_sig2_z = 1.0 / dat[:,4]**2.0

# We cannot use np.inf with cma-es so we use our custom definition of infinity
# Just be sure that inf is bigger than the fitness of the worst cases
inf = 1.0e100

# Checking definite positiveness with negative mass models imply searching the whole
# space for negative density. Here we use a trick to check the validity :
# We check the density on a grid of points up to a certain distance (1Mpc along R,
# 100 kpc along z). Note that it is important, once the model has been obtained
# to run a proper minimization on the MNnModel to check that the model is really
# positive definite. We don't do this in the loss function to to speed up the
# computation time
dP_R = np.logspace(0.0, 3.0, 10)
dP_z = np.logspace(0.0, 2.0, 10)
DPR, DPz = np.meshgrid(dP_R, dP_z)

# Scaling of the parameters. Try to scale the model so that the parameters
# are roughly of the same ordersd of magnitudes. Here, a and b are in kpc
# M is in 10^10 Msun
M_scale = 1.0e10

# Custom G, in kpc^3.Msun^-1.Myrs^-2 (consistent with McMillan 2017 and GalPot)
mnn.model.G = 4.49865897e-12

# Dummy array to fill in for the y component in the model
y = np.zeros(dat[:,0].shape[0])

# Loss function :
# We create a MNnModel, fill it with the valued of the disc
# Check all the priors, and return the cost : the residual
# on the forces of the model
def loss(discs):
    tmp_model = MNnModel()
    n = len(discs)
    tot_m = 0.0

    # Scaling and adding each disc. Checking for priors on
    # individual discs : a+b >= 0 and b > 0. We allow
    # negative a and M here.
    for i in range(n//3):
        a, b, M = discs[i*3:(i+1)*3]

        if a + b < 0:
            return inf

        if b < 0.0:
            return inf

        tot_m += M
        tmp_model.add_disc('z', a, b, M*M_scale)

    # The total mass of the model must be positive
    if tot_m < 0.0:
        return inf

    # Verifying if the model is definite-positive
    dp_test = tmp_model.evaluate_density(DPR, 0.0, DPz)
    if np.sum(dp_test < 0.0) > 0.0:
        return inf

    # Getting the forces and computing the residual
    forces = tmp_model.evaluate_force(dat[:,0], y, dat[:,1])

    fR = forces[0,:]
    fz = forces[2,:]

    res_R = np.sum((dat[:,3] - fR)**2.0 * inv_sig2_R)
    res_z = np.sum((dat[:,4] - fz)**2.0 * inv_sig2_z)

    return res_R + res_z

# Initial guess, be careful here : initial optimizer population
# is going to be sampled from a gaussian centered on x0 with
# std = sigma. it is important to make sure that no invalid
# values can be sampled from this or the optimizer will immediatly
# crash
x0    = [1.0]*ndiscs*3
sigma = 1.0e-1

f = loss(x0)
print('Base fitness = ', f)
if f >= inf:
    exit(0)

# Instantiation of the optimizer + run
es = cma.CMAEvolutionStrategy(x0, sigma,
                              {'tolconditioncov':1.0e20, # Relaxing the conditioning tolerance
                               'maxiter':1000000})       # 1 million iterations max
es.optimize(loss)

# Printing the best fit in a Python way. Remember that this is
# the scaled values. In theory all M should be multiplied by M_scale
print('Best fit : ')
print('[' + ','.join(str(x) for x in es.result[0]) + ']')
