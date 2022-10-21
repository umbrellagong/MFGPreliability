import sys
sys.path.append("../../")
import os
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from core import AcqIVR_FP, DiscreteInputs, OptimalDesign, failure_probability
from examples.cutin.cutin import value_function_IDM


def f_h(x):
    x = np.array(x)
    if x.ndim == 1:
        return value_function_IDM(x, dt=0.2, STEP=50)
    if x.ndim == 2:
        return np.array([value_function_IDM(i, dt=0.2, STEP=50) for i in x])


def main():
    
    # def trails 
    n_trails = 200

    # build the input
    dim = 2
    domain = np.array([[0, 90], [-20, 10]])
    
    raw_prob = np.loadtxt(open("cutin_table.csv", "rb"), 
                          delimiter=",", skiprows=1)
    range_rate = np.arange(-20, 10.1, 0.4)
    range_, raw_prob = raw_prob[:,0], raw_prob[:,1:]
    range_rata_mesh, range_mesh = np.meshgrid(range_rate , range_)
    grid = np.concatenate((range_mesh.reshape(-1,1), 
                                       range_rata_mesh.reshape(-1,1)), axis=1)
    weights = raw_prob.reshape(-1)
    
    inputs = DiscreteInputs(domain, dim)
    inputs.set_pdf(grid, weights)

    # build the kernel  # kernel
    kernel = (C(100, (1, 1e4)) * RBF((1e1, 1e1), (1, 5*1e2))
            + WhiteKernel(1e-2, (1e-3, 1e-1)))
    
    n_init = 16
    n_seq = 160 - n_init

    def wrapper_bm(trail):
        warnings.filterwarnings("ignore")
        sgp = GaussianProcessRegressor(kernel, normalize_y=False, 
                                       n_restarts_optimizer=6) 
        acq = AcqIVR_FP(inputs)
        np.random.seed(trail)
        opt = OptimalDesign(f_h, inputs)
        opt.init_sampling(n_init)
        models = opt.seq_sampling(n_seq, acq, sgp, n_jobs=1, 
                                  n_starters=30)
                                                   
        est = failure_probability(models, inputs)
        #return models, errors
        return est

    results = Parallel(n_jobs=10)(delayed(wrapper_bm)(j)
                                     for j in range(n_trails))
    np.save('results/sf', results)

if __name__=='__main__':
    main()