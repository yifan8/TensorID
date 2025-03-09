from .hadamard import hadamard
from .misc import numel
from numpy import log2

class KFJLT:
    # implement the KFJLT applied to a !!CP!! format
    # the CP factors MUST have #nrow = 2**int
    
    def __init__(self, backend = 'cpu', seed = None):
        self.backend = backend
        if backend == 'cpu':
            import numpy
            self.lib = numpy
        elif backend == 'gpu':
            try:
                import cupy
            except:
                raise ValueError(f'got backent = {backend} but gpu library cupy not available')
            self.lib = cupy
        else:
            raise ValueError('Backend must be eigher "cpu" or "gpu"')
        self.seed = seed
    
    
    def apply(self, factors, n_sample = None, seed = None, row_idx = None, unraveled = True, signs = None, to_cpu = False, overwrite = False):
        if seed is not None:
            self.lib.random.seed(seed)
        elif self.seed is not None:
            self.lib.random.seed(self.seed)
            
        if row_idx is None and n_sample is None:
            raise ValueError('must specify the number of rows as n_sample, or give the row indices for the sample')
        
        d = len(factors)
        ns = self.lib.array([len(factor) for factor in factors])
        N = self.lib.prod(ns)
        
        for (i,n) in enumerate(ns):
            if n & (n - 1) != 0:
                raise ValueError(f'Length of the factor {i} is {n:d}, not a power of 2')
        
        if row_idx is None:
            row_idx = self.lib.random.randint(low = 0, high = N, size = n_sample)
            unraveled = False
        else:
            # row_idx is given
            n_sample = row_idx.shape[-1]
        if not unraveled:
            row_idx_unrev = self.lib.array(self.lib.unravel_index(row_idx, ns), dtype = self.lib.int64)  # (d, n_sample)
        else:
            row_idx_unrev = row_idx
        
        if signs is None:
            signs = []
            for dim in range(d):
                signs.append(self.lib.random.choice([-1, 1], size = (int(ns[dim]), 1)))
        
        out = self.lib.full((n_sample, factors[0].shape[1]),
                             self.lib.sqrt(N/n_sample))
        for dim in range(d):
            out *= hadamard(signs[dim] * factors[dim], overwrite = overwrite)[row_idx_unrev[dim]]
        
        if self.backend == 'gpu' and to_cpu:
            out = out.get()
        return out
    
    
    def get_operator(self, n_sample, in_shapes, seed = None):
        if seed is not None:
            self.lib.random.seed(seed)
        elif self.seed is not None:
            self.lib.random.seed(self.seed)
            
        N = numel(in_shapes)
        for (i,n) in enumerate(in_shapes):
            if n & (n - 1) != 0:
                raise ValueError(f'Length of the factor {i} is {n:d}, not a power of 2')
        row_idx = self.lib.random.randint(low = 0, high = N, size = n_sample, dtype = self.lib.int64)
        row_idx_unrev = self.lib.array(self.lib.unravel_index(row_idx, in_shapes), dtype = self.lib.int64)  # (d, n_sample)
        signs = []
        for dim in range(len(in_shapes)):
            signs.append(self.lib.random.choice([-1., 1.], size = (in_shapes[dim], 1)))
        return row_idx_unrev, signs
    
    

        