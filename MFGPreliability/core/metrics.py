import numpy as np


def failure_probability(models, inputs, compensation=None):
    '''Compute failure probability.
    '''
    res = np.zeros(len(models))
    
    if models[0].X_train_.shape[1] != inputs.dim:
        grid = np.hstack((inputs.grid, np.ones((inputs.grid.shape[0],1))))
    else:
        grid = inputs.grid

    for i, model in enumerate(models):
        mu = model.predict(grid).flatten() 
        if compensation is not None:
            mu += compensation
        res[i] = np.average(mu < 0, weights=inputs.weights)
        
    return res
