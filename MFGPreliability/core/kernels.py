import numpy as np
from sklearn.gaussian_process.kernels import CompoundKernel
import copy



class TFkernel(CompoundKernel):
    '''
    A bi-fidelity kernel.
    
    Kernel for autogressive model where f_h = rho * f_l + d.

    parameter
    ---------
    A list of kernels including 
        rbf_l, rbf_d: instances of RBF_
        ll_l, ll_d: instances of Whitekernel
        exp_rho: instance of Constantkernel
            e^rho
    '''
    def __init__(self, kernels):
        self.rbf_l = kernels[0]
        self.rbf_d = kernels[1]
        self.ll_l = kernels[2]
        self.ll_h = kernels[3]
        self.exp_rho = kernels[4]
        # To use the theta properties to change and read hyperparameters
        self.kernels = kernels

    # The bounds in CompoundKernel are wrong. Need re-write.
    @property
    def bounds(self):
        bounds_ = np.empty((0,2))
        for kernel in self.kernels:
            if kernel.bounds.size != 0:     # unfixed only
                bounds_ = np.vstack((bounds_, kernel.bounds))
        return bounds_

    def __call__(self, X, Y=None, eval_gradient=False):
        '''
        Compute the kernel value and gradient w.r.t. hyperparameters.

        parameter
        ----------
        X: array (n1, d+1)
            augmented with the fidelity: 1 for high-fidelity and 0 for low.
        Y: array (n2, d+1)
            augmented with the fidelity: 1 for high-fidelity and 0 for low.

        return 
        ---------
        K: array (n1, n2)
            kernel value
        grad: array (n1, n2, d)
            gradient of kernel w.r.t. hyperparameters

        '''
        num_h_X = np.count_nonzero(X[:,-1]==1)    # Number of hf samples
        rho = np.log(self.exp_rho.constant_value) # The true rho
        X = np.copy(X)[:,:-1]                     # protect the original X
        if Y is None:                             
            if eval_gradient:  
                K_l, dK_rbf_l = self.rbf_l(X, eval_gradient=True)
                dK_rho = np.copy(K_l)
                K_d, dK_rbf_d_ = self.rbf_d(X[:num_h_X], eval_gradient=True)
                
                K_llh, dK_llh_= self.ll_h(X[:num_h_X], eval_gradient=True)
                K_lll, dK_lll_= self.ll_l(X[num_h_X:], eval_gradient=True)

                K = self.compute_K(K_l, K_d, rho, num_h_X, K_lll, K_llh)

                # always have the rbf_l and rbf_d unfixed
                dK_rbf_l[:num_h_X, :num_h_X] *= rho**2
                dK_rbf_l[:num_h_X, num_h_X:] *= rho
                dK_rbf_l[num_h_X:, :num_h_X] *= rho
                dK_rbf_d = np.zeros((X.shape[0], X.shape[0], 
                                     len(self.rbf_d.theta)))
                dK_rbf_d[:num_h_X, :num_h_X, :] = dK_rbf_d_

                # ll_l, ll_h and rho could be fixed and their derivative would
                # be empty.
                if self.ll_h.hyperparameters[0].fixed:
                    dK_llh = np.empty((X.shape[0], X.shape[0], 0))
                else:
                    dK_llh = np.zeros((X.shape[0], X.shape[0], 1))
                    dK_llh[:num_h_X, :num_h_X, 0] = dK_llh_

                if self.ll_l.hyperparameters[0].fixed:
                    dK_lll = np.empty((X.shape[0], X.shape[0], 0))
                else:
                    dK_lll = np.zeros((X.shape[0], X.shape[0], 1))
                    dK_lll[num_h_X:, num_h_X:, 0] = dK_lll_
                
                if self.exp_rho.hyperparameters[0].fixed:
                    dK_rho = np.empty((X.shape[0], X.shape[0], 0))
                else: 
                    dK_rho[:num_h_X, :num_h_X] *= 2*rho
                    dK_rho[num_h_X:, num_h_X:] = 0
                    dK_rho = dK_rho[:,:,np.newaxis]
                # assemble the derivatives
                dK_hyp = np.concatenate([dK_rbf_l, dK_rbf_d, 
                                         dK_lll, dK_llh, dK_rho], axis=2)
                return K, dK_hyp

            else:
                # No derivative needed
                K_l = self.rbf_l(X)             
                K_d = self.rbf_d(X[:num_h_X])   
                K_llh = self.ll_h(X[:num_h_X])  
                K_lll = self.ll_l(X[num_h_X:])
                K = self.compute_K(K_l, K_d, rho, num_h_X, K_lll, K_llh)
                return K
        else:
            if eval_gradient:
                raise('wrong!')
            num_h_Y = np.count_nonzero(Y[:,-1]==1)
            Y = np.copy(Y)[:,:-1]

            K_l = self.rbf_l(X, Y)
            K_d = self.rbf_d(X[:num_h_X], Y[:num_h_Y])
            K = self.compute_K(K_l, K_d, rho, num_h_X, num_h_Y=num_h_Y)
            return K

    def diag(self, X): 
        num_h_X = np.count_nonzero(X[:,-1]==1)
        rho = np.log(self.exp_rho.constant_value)
        X = np.copy(X)[:,:-1]
        diag_vec = self.rbf_l.diag(X)
        diag_vec[:num_h_X] *= rho**2
        diag_vec[:num_h_X] += self.rbf_d.diag(X[:num_h_X])
        diag_vec[:num_h_X] += self.ll_h.diag(X[:num_h_X])
        diag_vec[num_h_X:] += self.ll_l.diag(X[num_h_X:])
        return diag_vec


    @staticmethod
    def compute_K(K_l, K_d, rho, num_h_X, K_lll=0, K_llh=0, num_h_Y=None):
        if num_h_Y==None:
            num_h_Y = num_h_X
        
        K = K_l
        K[:num_h_X, :num_h_Y] *= rho**2                  
        K[:num_h_X, :num_h_Y] += K_d
        K[:num_h_X, :num_h_Y] += K_llh
        K[:num_h_X, num_h_Y:] *= rho
        K[num_h_X:, :num_h_Y] *= rho
        K[num_h_X:, num_h_Y:] += K_lll
        return K

    @property
    def theta(self):   # unfixed hyper-parameters
        return np.hstack([kernel.theta for kernel in self.kernels])

    @theta.setter
    def theta(self, theta):
        current_dims = 0
        for kernel in self.kernels:
            k_dims = kernel.n_dims
            kernel.theta = theta[current_dims: current_dims + k_dims]
            current_dims += k_dims

    def _check_bounds_params(self):
        pass

