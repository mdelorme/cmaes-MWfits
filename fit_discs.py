import matplotlib as mpl
mpl.use('pdf')
from mnn.fitter import MNnFitter
from mnn.model import MNnModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.optimize import minimize
import sys
import cma

ndiscs = int(sys.argv[1]) if len(sys.argv) > 1 else 6
print(' ===== Fitting Discs models from McMillan 2016 with {} MNn discs ====='.format(ndiscs))

dat  = np.loadtxt('../../data/rho_disc_restricted.dat')
yerr = 0.01*dat[:,3]

inf = 1.0e20 # We can't use numpy's infinite with CMA-ES

def fitness(discs):
    tmp_model = MNnModel()
    n = len(discs)
    tot_m = 0.0
    for i in range(n//3):
        a, b, M = discs[i*3:(i+1)*3]

        if a + b < 0:
            return inf

        if b < 0.0:
            return inf

        #if abs(a) > 1.0e6:
        #    return inf

        #if b > 1.0e6:
        #    return inf

        tot_m += M*1.0e9
        tmp_model.add_disc('z', a, b, M*1.0e9)

        
    if tot_m < 0.0:
        return inf

    if not tmp_model.is_positive_definite(1.0e6):
        return inf

    model = tmp_model.evaluate_density(dat[:,0], dat[:,1], dat[:,2])
    inv_sigma2 = 1.0/(yerr**2.0)
    return 0.5*(np.sum((dat[:,3]-model)**2.0*inv_sigma2))

es = cma.CMAEvolutionStrategy([1.0e3]*ndiscs*3, 0.5,
                              {'tolfacupx':50000,
                               'tolconditioncov':1.0e20,
                               'maxiter':50000})
es.optimize(fitness)

print('Best fit : ')
print(es.result[0])
