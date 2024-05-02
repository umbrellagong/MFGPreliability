import numpy as np
from scipy.stats import norm
from scipy.linalg import solve_triangular



class Acq(object):
 
    def __init__(self, inputs):
        self.inputs = inputs

    def compute_value(self, x):
        raise NotImplementedError

    def update_prior_search(self, model):     
        raise NotImplementedError


class AcqIVR_FP(Acq):

    def compute_value_tf_cost(self, pos, fidelity, cost):
        
        x = np.append(pos, fidelity)
        value = self.compute_value(x)
        return value + np.log(cost)

    def compute_value(self, x):
        
        x = np.atleast_2d(x)
        
        K_trans_x = self.model.kernel_(x, self.model.X_train_)
        
        V_x = solve_triangular(
                self.model.L_, K_trans_x.T, lower=True, check_finite=False)
        cov_x_grid = self.model.kernel_(x, self.grid) - V_x.T @ self.V_grid
        var_reduction = cov_x_grid ** 2 / ((self.model.predict(x, 
                                                        return_std=True)[1])**2)
        # compute the new exceeding probability of grid
        # assume the limit is 0
        P = norm.cdf(self.mean / np.sqrt(np.clip(self.std**2 - var_reduction, 
                                                 1e-10, None))) 
        V = P * (1-P)
        return - np.log(np.clip(self.U - np.sum(V * self.inputs.weights), 
                                1e-10, None))
    
    def update_prior_search(self, model, compensation=None):

        self.model = model
        if model.X_train_.shape[1] != self.inputs.dim:
            self.grid = np.hstack((self.inputs.grid, 
                        np.ones((self.inputs.grid.shape[0],1))))
        else:
            self.grid = self.inputs.grid
        
        self.mean, self.std = self.model.predict(self.grid, return_std=True)
        if compensation is not None:
            self.mean += compensation
            
        self.std = np.clip(self.std, 1e-10, None)
        P = norm.cdf(self.mean / self.std) 
        V = P * (1-P)
        self.U = np.sum(V * self.inputs.weights)
        
        K_trans_grid = self.model.kernel_(self.grid, self.model.X_train_)
        self.V_grid = solve_triangular(
                self.model.L_, K_trans_grid.T, lower=True, check_finite=False)