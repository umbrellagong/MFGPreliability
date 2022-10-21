import numpy as np

def f_h(X):
    if X.ndim==1:
        return ((1.5+X[0])**2+4)*((2.5+X[1])-1)/20-np.sin(5*(1.5+X[0])/2)-2
    if X.ndim==2:
        return (((1.5+X[:,0])**2+4)*((2.5+X[:,1])-1)/20-
                                                np.sin(5*(1.5+X[:,0])/2)-2)