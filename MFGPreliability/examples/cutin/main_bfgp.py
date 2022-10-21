import sys
sys.path.append("../../")
import warnings
import os

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel as C
from joblib import Parallel, delayed
from core import TFkernel, AcqIVR_FP, DiscreteInputs, OptimalDesignTF, failure_probability
from examples.cutin.cutin import value_function_IDM


def f_h(x):
    x = np.array(x)
    if x.ndim == 1:
        return value_function_IDM(x, dt=0.2, STEP=50) - 3
    if x.ndim == 2:
        return np.array([value_function_IDM(i, dt=0.2, STEP=50) for i in x]) - 3


def f_l(x):
    x = np.array(x)
    if x.ndim == 1:
        return value_function_IDM(x, dt=2, STEP=5) - 3
    if x.ndim == 2:
        return np.array([value_function_IDM(i, dt=2, STEP=5) for i in x]) - 3


def main(n_seq, n_cost, c_ratio):

    # def number of trails
    n_trails = 50

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

    # def kernel 
    kernel_l = C(100, (1, 1e4)) * RBF((1e1, 1e1), (1, 5*1e2))
    kernel_d = C(10, (1e-2, 1e3)) * RBF((1e2, 1e2), (1e-1, 5*1e2))
    likelihood_l = WhiteKernel(1e-6, 'fixed')
    likelihood_h = WhiteKernel(1e-10, 'fixed')
    exp_rho = C(np.exp(1), 'fixed')
    tfkernel = TFkernel([kernel_l, kernel_d, likelihood_l, 
                                                     likelihood_h, exp_rho])

    # def initial samples
    n_init_h, n_init_l = 8, 8 * c_ratio
    
    # generate results for each random seed
    def wrapper_bm(trail):
        warnings.filterwarnings("ignore")
        tfgp = GaussianProcessRegressor(tfkernel, n_restarts_optimizer=6)
        acq = AcqIVR_FP(inputs)
        np.random.seed(trail)
        opt = OptimalDesignTF(f_h, f_l, inputs)
        opt.init_sampling(n_init_h, n_init_l)

        models = opt.seq_sampling_opt(n_seq, n_cost, c_ratio, acq, tfgp, 
                                                n_starters=20, n_jobs=1)
        est = failure_probability(models, inputs)
        datas = [model.X_train_ for model in models]        
        
        return datas, est 
    
    results = Parallel(n_jobs=10)(delayed(wrapper_bm)(j)
                                          for j in range(n_trails))
    np.save('results/bf_5', results)


if __name__=='__main__':
    main(n_seq=600, n_cost=60, c_ratio=10)

    