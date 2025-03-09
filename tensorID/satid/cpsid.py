
from ..util import KFJLT

class CPSolver:
    
    def __init__(self, backend: str = 'cpu'):
        """SatID solver for CP tensor

        Parameters
        ----------
        backend : str, optional
            'cpu' or 'gpu', if 'gpu' then library cupy is required, by default 'cpu'

        Raises
        ------
        ValueError
            If backend == 'gpu' but cupy is not available
        """
        
        self.backend = backend
        if backend == 'cpu':
            import numpy
            self.lib = numpy
        else:
            try:
                import cupy
            except:
                raise ValueError(f'got backend = {backend} but gpu library cupy not available')
            self.lib = cupy
    
    
    def fit(self, factors: list, ranks: list[int], 
            axis: list[int] = None, 
            selection_rule: int = 1,
            sketch_dim: None | int | list[int] = None, 
            seed: None | int = None) -> tuple[list, list, list]:
        """Compute the SatID

        Parameters
        ----------
        factors : list
            list of CP factors, of shape (ni, r), i = 1,...,d
        ranks : list[int]
            target rank of CoreID   
        axis : list[int], optional
            axis from which the indices are selected, e.g., [2, 0] for a 3rd order tensor means first select from axis 2 then axis 0, no selection is made on axis 1, if None, this defaults to [0, ..., d-1], by default None
        selection_rule : int, optional
            selection rule, either 0 or 1, if 0 the uniform sampling of columns is used, if 1, norm sampling is used, by default 1
        sketch_dim : None | int | list[int], optional
            sketching dimension, if None, no sketching is used; if int, then use this sketching dimension for all modes; if list, then it contains the sketching dimension of each mode, in the same order as axis, by default None
        seed : None | int, optional
            random seed, by default None

        Returns
        -------
        tuple[list, list, list]
            1. index sets on each mode,
            2. satellite matrices on each mode,
            3. a new list of CP factors whose contraction gives the reconstructed tensor
        """
        
        if seed is not None:
            self.lib.random.seed(seed)
        nr = len(ranks)
        p = len(factors[0][1])
        if axis is None:
            axis = [i for i in range(nr)]
        idx_sets = [None for _ in range(nr)]
        satellites = [None for _ in range(nr)]
        d = len(factors)
        all_axes = axis.copy()
        for i in range(d):
            if i not in axis:
                all_axes.append(i)

        use_sketch = sketch_dim is not None
        if use_sketch:
            if isinstance(sketch_dim, int):
                sketch_dim = [sketch_dim for _ in range(nr)]
            
        for i in range(nr):
            cur_mode = axis[i]
            A = factors[cur_mode]
            Ahat = A.copy()
            r = ranks[i]
            n = A.shape[0]
            m = sketch_dim[i]
            idx_modes = [all_axes[j] for j in range(d) if j != i]
            
            J = []
            sate = self.lib.zeros((r, n))
            
            for k in range(r):
                w = self.lib.ones(p)
                bs = self.lib.zeros(d-1, dtype = self.lib.int32)
                for s in range(d-1):
                    # sample index
                    if selection_rule == 1:
                        # use norm sampling
                        if use_sketch:
                            if s < d-2:
                                S = KFJLT(backend = self.backend)
                                # apply sketch if we have at least 2 nodes left
                                M = S.apply([factors[idx_modes[t]] for t in range(s+1, d-1)] + [Ahat], n_sample = m)
                                M *= w
                                M = M @ factors[idx_modes[s]].T
                            else:
                                # we brute force it
                                M = Ahat * w @ factors[idx_modes[s]].T
                        
                            score = self.lib.linalg.norm(M, axis = 0)**2
                            score /= self.lib.sum(score)
                        else:
                            core = (temp := Ahat * w).T @ temp  # (p, p)
                            for t in range(s+1, d-1):
                                core *= factors[idx_modes[t]].T @ factors[idx_modes[t]]
                            score = self.lib.einsum('ix, xy, iy -> i', factors[idx_modes[s]], core, factors[idx_modes[s]], optimize = 'optimal')
                            score = self.lib.maximum(score, 0)
                            score /= self.lib.sum(score)
                        bs[s] = self.lib.random.choice(self.lib.arange(n), size = 1, p = score)
                    elif selection_rule == 0:
                        # use uniform
                        bs[s] = self.lib.random.choice(self.lib.arange(n), size = 1)
                    else:
                        raise ValueError(f'method {selection_rule} not understood, must be 0 or 1')
                    
                    w *= factors[idx_modes[s]][bs[s]]
                
                # row index already computed
                sate[k] = A @ w
                J.append(bs)
                
                # update Ahat
                q = Ahat @ w
                q /= self.lib.linalg.norm(q)
                Ahat -= self.lib.outer(q, (Ahat.T @ q))
            
            # save selected nodes
            satellites[i] = sate
            idx_sets[i] = J
        
        new_factors = []
        for i in range(len(factors)):
            # sates[i] has shape (rank, n)
            Q, _ = self.lib.linalg.qr(satellites[i].T, mode = 'reduced')
            new_factors.append(Q @ (Q.T @ factors[i]))  
            
        return idx_sets, satellites, new_factors

                    
                    
                    
                        