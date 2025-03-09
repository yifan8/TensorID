## solver for the CP CID problem using sketch


from ..util import KFJLT, NormBased, Nuclear


class CPSolver:
    
    letters = 'abcdefghijklmnopqrstuvwxyz'
    
    def __init__(self, backend: str = 'cpu'):
        """CoreID solver for CP tensor

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
            selection_rule: str | int = 'inf',
            sketch_dim: None | int | list[int] = None, 
            seed: None | int = None, 
            overwrite: bool = False) -> tuple[list, list, list]:
        """Compute the CoreID

        Parameters
        ----------
        factors : list
            list of CP factors, of shape (ni, r), i = 1,...,d
        ranks : list[int]
            target rank of CoreID   
        axis : list[int], optional
            axis from which the indices are selected, e.g., [2, 0] for a 3rd order tensor means first select from axis 2 then axis 0, no selection is made on axis 1, if None, this defaults to [0, ..., d-1], by default None
        selection_rule : str | int, optional
            selection rule, 'inf', 'nuc' or a positive number f, for which a sampling is made according to pmf = column norms to power f, by default 'inf'
        sketch_dim : None | int | list[int], optional
            sketching dimension, if None, no sketching is used; if int, then use this sketching dimension for all modes; if list, then it contains the sketching dimension of each mode, in the same order as axis, by default None
        seed : None | int, optional
            random seed, by default None
        overwrite : bool, optional
            if True, then the output will overwrite the input array factors, by default False

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
        d = len(factors)
        if axis is None:
            axis = [i for i in range(nr)]
        idx_sets = [None for _ in range(nr)]
        satellites = [None for _ in range(nr)]

        use_sketch = sketch_dim is not None
        if use_sketch and isinstance(sketch_dim, int):
            sketch_dim = [sketch_dim for _ in range(nr)]
        
        if not overwrite:
            new_factors = [x for x in factors]  # store X @ factors after selection
        else:
            new_factors = factors
            
        if selection_rule == 'nuc':
            solver = Nuclear(self.backend)
        else:
            solver = NormBased(self.backend)
        
        if not use_sketch:
            signature = ','.join([self.letters[i] + 'X' for i in range(len(factors))]) + '->' + self.letters[:len(factors)]
            tensor = self.lib.einsum(signature, *factors, optimize = 'optimal')
        
        for i in range(nr):
            transform_factors = [new_factors[j] for j in range(d) if j != axis[i]]
            if use_sketch:
                S = KFJLT(backend = self.backend)
                B = S.apply(transform_factors, n_sample = sketch_dim[i])  # (m, p)
                B = B @ new_factors[axis[i]].T  # (m, n)
            else:
                target = ''.join([self.letters[j] for j in range(d) if j != axis[i]]) + self.letters[axis[i]]
                B = self.lib.einsum(self.letters[:d] + '->' + target, tensor).reshape(-1, new_factors[axis[i]].shape[0])
            
            J, X = solver.select(B, ranks[i], rule = selection_rule)
            X *= X[0, J[0]]
            
            new_factors[axis[i]] = X.T @ new_factors[axis[i]][J]   
            if not use_sketch:
                tensor = self.lib.take(tensor, J, axis = axis[i])
                out = self.letters[:axis[i]] + 'X' + self.letters[axis[i]+1:d]
                signature = self.letters[:d] + ',' + self.letters[axis[i]] + 'X ->' + out
                tensor = self.lib.einsum(signature, tensor, X)
            idx_sets[i] = J
            satellites[i] = X
        
        return idx_sets, satellites, new_factors
        
        
        