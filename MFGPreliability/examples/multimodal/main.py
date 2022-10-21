import sys
sys.path.append("../../")
import os
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from core import AcqIVR_FP, GaussianInputs, OptimalDesign, failure_probability
from multimodal import f_h

def main():    
    
    # def trails
    n_trails = 100

    # build the input
    num = 10000
    dim = 2
    mean = np.zeros(2)
    cov = np.eye(2)
    domain = np.array([[-6,6]]*dim)
    inputs = GaussianInputs(domain, dim)
    np.random.seed(0)
    inputs.set_pdf(mean, cov, num, uniform=False)

    # build the kernel
    kernel = C(100, (1e-1,1e3)) * RBF((1e1,1e1), (1e-1, 1e2))
    
    # sequential details
    n_init=8
    n_seq=22

    def wrapper_bm(trail):
        warnings.filterwarnings("ignore")
        sgp = GaussianProcessRegressor(kernel, normalize_y=False, 
                                       n_restarts_optimizer=6)
        np.random.seed(trail)
        acq = AcqIVR_FP(inputs)
        opt = OptimalDesign(f_h, inputs)
        opt.init_sampling(n_init)
        models = opt.seq_sampling(n_seq, acq, sgp, n_jobs=1, n_starters=20)

        est = failure_probability(models, inputs)
        #return models, est
        return est

    results = Parallel(n_jobs=10)(delayed(wrapper_bm)(j) 
                                  for j in range(n_trails))
    np.save('results', results)



if __name__=='__main__':
    main()