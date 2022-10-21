import numpy as np
from scipy.stats import multivariate_normal
from pyDOE import lhs
        

class DiscreteInputs(object): 
    
    def __init__(self, domain, dim):
        
        self.domain = domain
        self.dim = dim
        
    def sampling(self, num, criterion=None):
        
        lhd = lhs(self.dim, num, criterion=criterion)
        lhd = self.rescale_samples(lhd, self.domain)
        return lhd
    
    def set_pdf(self, grid, weights): 
        'Discrete distribution'
        self.grid = grid
        self.weights = weights

    @staticmethod
    def rescale_samples(x, domain):
        """Rescale samples from [0,1]^d to actual domain."""
        for i in range(x.shape[1]):
            bd = domain[i]
            x[:,i] = x[:,i]*(bd[1]-bd[0]) + bd[0]
        return x    


class GaussianInputs(DiscreteInputs): 
    def set_pdf(self, mean, cov, num, uniform=True):
    
        if uniform:
            x, y =  np.meshgrid(*[np.linspace(bd[0], bd[1], int(np.sqrt(num)+1))
                                   for bd in self.domain])
            self.grid = np.concatenate((x.reshape(-1,1), 
                                        y.reshape(-1,1)), axis=1)
            self.weights = multivariate_normal(mean, cov).pdf(self.grid) 
        else:
            self.grid = np.random.multivariate_normal(mean, cov, num)
            self.weights = np.ones(num) / num
            
        