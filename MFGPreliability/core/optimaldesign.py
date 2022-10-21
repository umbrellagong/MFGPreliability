import copy
import numpy as np
from scipy import optimize
from sklearn.base import clone
from joblib import Parallel, delayed
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array
from .metrics import failure_probability


class OptimalDesign(object):

    def __init__(self, f, inputs):
        self.f = f
        self.inputs = inputs

    def init_sampling(self, n_init): 
        self.DX = self.inputs.sampling(n_init)
        self.DY = self.f(self.DX)   # a vector
        return self

    def seq_sampling(self, n_seq, acq, model, n_jobs=6, compensation=None, 
                           discrete=False, n_starters=30, verbose=True): 
        self.acq = copy.copy(acq)
        self.model = clone(model)
        self.model_list = []
        
        for i in range(n_seq):
            self.model.fit(self.DX, self.DY)
            self.model_list.append(copy.deepcopy(self.model))
            self.acq.update_prior_search(self.model, compensation)
            
            if discrete:
                res = Parallel(n_jobs=n_jobs)(delayed(self.acq.compute_value)
                      (i) for i in range(self.inputs.grid.shape[0]))
                opt_pos = self.inputs.grid[np.argmin(res)]
                #      (i) for i in candidates)
                #opt_pos = candidates[np.argmin(acq_values)]
            else:
                init = self.inputs.sampling(n_starters)
                res = Parallel(n_jobs=n_jobs)(delayed(optimize.minimize)
                                                (self.acq.compute_value,
                                                init[j], method="L-BFGS-B",
                                                bounds = self.inputs.domain,
                                                jac = None, 
                                                options={'gtol': 1e-3})
                                                for j in range(init.shape[0]))
                opt_pos = res[np.argmin([k.fun for k in res])].x
            self.DX = np.append(self.DX, np.atleast_2d(opt_pos), axis=0)
            self.DY = np.append(self.DY, self.f(opt_pos))
            if verbose:
                with open('progress.o', 'a') as f:
                        f.write(str(i) + '\n')
            print(i)

        # train the last model
        self.model.fit(self.DX, self.DY)
        self.model_list.append(copy.deepcopy(self.model))

        return self.model_list
        
        
class OptimalDesignTF(object):        

    def __init__(self, f_h, f_l, inputs):
        self.f_h = f_h
        self.f_l = f_l
        self.inputs = inputs
        
    def load_data(self, DX):
        ''' Start from existing dataset DX.
        '''
        idx_low = np.where(DX[:,2]==0)[0][0]
        DX_h = DX[:idx_low][:,:2]
        DY_h = self.f_h(DX_h)
        DX_l = DX[idx_low:][:,:2]
        DY_l = self.f_l(DX_l)

        self.DY = np.append(DY_h, DY_l) 
        self.DX = DX  

        return self


    def init_sampling(self, n_init_h, n_init_l): 
        '''Generate initial samples.
        
        Parameters
        -----------
        n_init_h, n_init_l: int
            number of high and low-fidelity initial samples
        '''
        DX_h = self.inputs.sampling(n_init_h)
        DY_h = self.f_h(DX_h)

        DX_l = self.inputs.sampling(n_init_l)
        DY_l = self.f_l(DX_l)

        DX = convert_x_list_to_array([DX_l, DX_h])
        DY = np.append(DY_l, DY_h)
        self.DX = np.flip(DX, axis=0) 
        self.DY = np.flip(DY)
        return self
        

    def seq_sampling_opt(self, n_seq, n_cost, c_ratio, acq, model, 
                         n_jobs=1, n_starters=20, verbose=True): 

        self.acq = copy.deepcopy(acq)
        self.model = copy.deepcopy(model)
        self.model_list = []

        for ii in range(n_seq):
            self.model.fit(self.DX, self.DY)
            self.model_list.append(copy.deepcopy(self.model))
            self.acq.update_prior_search(self.model)
            init = self.inputs.sampling(n_starters)
            res_l = Parallel(n_jobs=n_jobs)(delayed(optimize.minimize)
                                            (self.acq.compute_value_tf_cost,
                                                init[j], 
                                                args=(0, 1), # low-fidelity
                                                method="L-BFGS-B",
                                                jac=False,
                                                bounds = self.inputs.domain,
                                                options={'gtol': 1e-3})
                                            for j in range(init.shape[0]))
            self.res_l = res_l  
            opt_pos_l = res_l[np.argmin([k.fun for k in res_l])].x
            opt_value_l = res_l[np.argmin([k.fun for k in res_l])].fun
            res_h = Parallel(n_jobs=n_jobs)(delayed(optimize.minimize)
                                            (self.acq.compute_value_tf_cost,
                                                init[j], 
                                                args=(1, c_ratio),
                                                method="L-BFGS-B",
                                                jac=False,
                                                bounds = self.inputs.domain,
                                                options={'gtol': 1e-3})
                                            for j in range(init.shape[0]))
            self.res_h = res_h  
            opt_pos_h = res_h[np.argmin([k.fun for k in res_h])].x
            opt_value_h = res_h[np.argmin([k.fun for k in res_h])].fun

            if opt_value_h < opt_value_l: # high-fidelity sampling!
                self.DX = np.insert(self.DX, 0, 
                                    np.append(opt_pos_h, 1), axis=0)
                self.DY = np.insert(self.DY, 0, self.f_h(opt_pos_h)) 
                print(ii, '  ', np.append(opt_pos_h, 1))     
            else:
                self.DX = np.insert(self.DX, self.DX.shape[0], 
                                    np.append(opt_pos_l, 0), axis=0)
                self.DY = np.append(self.DY, self.f_l(opt_pos_l))
                print(ii, '  ', np.append(opt_pos_h, 0))     

            num_h_X = np.count_nonzero(self.DX[:,-1]==1)
            num_l_X = np.count_nonzero(self.DX[:,-1]==0)
            cost = num_h_X + 1 / c_ratio * num_l_X
            
            if verbose:
                with open('progress_bf.o', 'a') as f:
                        f.write(str(ii) + ' ' + str(cost) + '\n')
                        
            if cost > n_cost:
                break
            
        self.model.fit(self.DX, self.DY)
        self.model_list.append(copy.deepcopy(self.model))
        return self.model_list
