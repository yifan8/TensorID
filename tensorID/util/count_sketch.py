## implementation of count sketch
## for a map from R^(n1.n2...nd) to Rm
## define a random tensor T having shape n1, ..., nd
## the entries in T are iid uniform {\pm 1, ..., \pm m}
## for every v in the input tensor with index (i1,...,id)
## update the sketch result y += e_{|T[i1,...,id]|} * sign(T[i1,...,id]) * v
## to apply this map efficiently, we generate T as the information tensor
## and define and store a CSR matrix S constructed from T
## S[|T[i1,...,id]|, (i1,...,id)] = sign(T[i1,...,id])

import numpy as np
import scipy.sparse as sps

cupy_not_available = False
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsps
except:
    cupy_not_available = True

class CS:
    
    def __init__(self, backend = 'cpu'):
        self.backend = backend
        if backend == 'cpu':
            self.lib = np
            self.sp = sps
        else:
            if cupy_not_available:
                raise ValueError('gpu library cupy not available')
            self.lib = cp
            self.sp = cpsps
        self.S = None
    
    
    def gen_apply(self, A, out_dim, seed = None):
        # require A is csr format
        shape = (out_dim, A.shape[0])
        if isinstance(A, self.sp.csc_matrix):
            self.generate(shape, seed, 'csc')
        else:
            self.generate(shape, seed, 'csr')
        out = self.apply(A)
        return out  # return CSR
    
    
    def generate(self, shape, seed = None, format = 'csr'):
        if seed is not None:
            self.lib.random.seed(seed)
        hi = self.lib.random.choice(shape[0], size = shape[1])
        data = self.lib.random.choice([-1., 1.], size = shape[1])
        if format == 'csr':
            S = self.sp.csr_matrix(
                (data, (hi, self.lib.arange(shape[1]))),
                shape = shape
                )
        else:
            S = self.sp.csc_matrix(
                (data, (hi, self.lib.arange(shape[1]))),
                shape = shape
                )
        self.S = S
    
    
    def apply(self, A):
        return self.S @ A
            


