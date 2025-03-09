## fast Hadamard transform

import numpy as np

def hadamard(x, axis = 0, overwrite = False):
    d = x.ndim
    if x.shape[axis] & (x.shape[axis] - 1) != 0:
        raise ValueError(f'Length {x.shape[axis]:d} is not a power of 2')
    if axis >= d:
        raise ValueError(f'Axis {axis:d} is greater than the dimension of the input')
    loglen = int(np.log2(x.shape[axis] + 0.1))
    outshape = x.shape
    newshape = list(x.shape)[:axis] + [2] * loglen + list(x.shape)[axis+1:]
    x = x.reshape(newshape)
    
    if not overwrite:
        out = x.copy()
    else:
        out = x
    
    ind = [slice(None)] * (d + loglen - 1)
    for ax in range(loglen):
        ind0 = ind.copy()
        ind1 = ind.copy()
        ind0[axis + ax] = 0
        ind1[axis + ax] = 1
        ind0 = tuple(ind0)
        ind1 = tuple(ind1)
        
        out[ind0] += out[ind1]
        out[ind1] *= -2
        out[ind1] += out[ind0]
        
    out /= np.sqrt(2)**(loglen)
    return out.reshape(outshape)
        

    
    
    