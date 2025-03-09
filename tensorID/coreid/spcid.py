## solver for the Sparse CID problem using sketch

from ..util import KFJLT, CS, hadamard, Nuclear, NormBased, SpTensor, next_pow2, numel

from collections.abc import Iterable


class SparseSolver:
    LETTERS = 'abcdefghijklmnopqrstuvwxyz'
    
    def __init__(self, backend = 'cpu'):
        """CoreID solver for sparse tensor

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
                raise ValueError(f'got backent = {backend} but gpu library cupy not available')
            self.lib = cupy
        
        
    
    def fit(self, T: SpTensor, ranks: list[int], 
            axis: list[int] = None, 
            selection_rule: str | int = 'inf',
            sketch_dim: None | int | list[int] = None, 
            seed: None | int = None) -> tuple[list, list]:
        """Compute the CoreID

        Parameters
        ----------
        T : SpTensor
            the sparse tensor to be decomposed
        ranks : list[int]
            target rank of CoreID   
        axis : list[int], optional
            axis from which the indices are selected, e.g., [2, 0] for a 3rd order tensor means first select from axis 2 then axis 0, no selection is made on axis 1, if None, this defaults to [0, ..., d-1], by default None
        selection_rule : str | int, optional
            selection rule, 'inf', 'nuc' or a positive number f, for which a sampling is made according to pmf = column norms to power f, by default 'inf'
        sketch_dim : None | tuple[ int|list[int], int|list[int], int|list[int] ], optional
            sketching dimension, if None, no sketching is used; if a tuple of 3 is used, then they corresponds to sketching dimension of KFJLT, count sketch, and the final sketch, each entry in the tuple can be either int (same sketching dimension is used for all axes) or a list of int (specifying the sketching dimesnion for each axis), by default None
        seed : None | int, optional
            random seed, by default None

        Returns
        -------
        tuple[list, list]
            1. index sets on each mode,
            2. satellite matrices on each mode
        """
        
        #! require the input T is pyttb.COO
        if seed is not None:
            self.lib.random.seed(seed)
        
        #! first process the tensor to a convenient view
        nr = len(ranks)
        d = len(T.shape)
        if axis is None:
            axis = [i for i in range(nr)]
            order = [i for i in range(nr)]
        else:
            added = [False for _ in range(d)]
            order = [0 for _ in range(d)]
            new_ranks = [0 for _ in range(nr)]
            for (i, m) in enumerate(axis):
                added[m] = True
                order[i] = m
                new_ranks[i] = ranks[m]
            cursor = len(axis)
            ranks = new_ranks
            for i in range(d):
                if not added[i]:
                    order[cursor] = i
                    cursor += 1
        T = T.transpose(order)  #[axis0, axis1, ..., axis(nr-1), rest...]
        shape = T.shape
        #ns = shape[:nr]
        
        #! these are return values
        idx_sets = [None for _ in range(nr)]
        satellites = [None for _ in range(nr)]
        Rs = [None for _ in range(nr)]

        #! sketch dimensions
        use_sketch = sketch_dim is not None
        if use_sketch:
            kf = KFJLT(backend = T.device)
            cs = CS(backend = T.device)
            kf_dim, cs_dim, out_dim = sketch_dim
            if isinstance(kf_dim, int):
                kf_dim = [kf_dim for _ in range(nr)]
            if isinstance(cs_dim, int):
                cs_dim = [cs_dim for _ in range(nr)]
            if isinstance(out_dim, int):
                out_dim = [out_dim for _ in range(nr)]
        
        #! matrix algorithm
        if selection_rule == 'nuc':
            solver = Nuclear(backend = T.device)
        else:
            solver = NormBased(backend = T.device)
        
        #! start adaptive selection over nr modes
        for i in range(nr):
            #! first compute the sketch A
            r_kf = numel(ranks[:i])
            n_sel = shape[i]
            A = T.reshape(r_kf, n_sel, -1)
            A = A.transpose(0, 2, 1)  # r_kf, n_cs, n_sel
            A.prune(axis = 1)
            n_cs = A.shape[1]
            
            if use_sketch and n_cs > 5 * cs_dim[i]:
                # there is benefit to sketch
                A = A.transpose(1, 0, 2)
                A = A.reshape(n_cs, r_kf * n_sel)
                #A.prune(axis = 0)
                A = A.to_sparse_matrix(A.sps.csc_matrix)  
                A = cs.gen_apply(A, cs_dim[i])  # sps.csc
                A = T.from_sparse_matrix(A)
                A = A.reshape(cs_dim[i], r_kf, n_sel)
                A = A.transpose(1, 0, 2)
            
            # A is sptensor
            if use_sketch and r_kf > 5 * kf_dim[i]:
                # there is benefit to sketch, only put the L factor into T
                A = A.reshape(list(ranks[:i]) + [n_cs, n_sel])
                in_shapes = [next_pow2(ranks[j]) for j in range(i)]
                row_idx, signs = kf.get_operator(kf_dim[i], in_shapes)
                factors = []
                for dim in range(i):
                    mat = T.lib.zeros((in_shapes[dim], ranks[dim]))
                    mat[:ranks[dim]] = signs[dim][:ranks[dim]] * Rs[dim]
                    factors.append(hadamard(mat)[row_idx[dim]].T)
                A = A.mttkrp(factors, fold_back = True)  # sptensor or ndarray, (n_cs, n_sel, kf_dim)
                A = A.transpose(2, 0, 1)  # kf_dim, n_cs, n_sel
                A *= self.lib.sqrt(r_kf / kf_dim[i])
            elif i > 0:
                # merge Ri into A
                A = A.reshape(list(ranks[:i]) + [-1, n_sel])  # SpTensor
                A = A.ttm(Rs[:i])
                
            A = A.reshape(-1, n_sel)  # sptensor
            if isinstance(A, SpTensor):
                A = A.to_sparse_matrix(T.sps.csc_matrix)
            
            # A is either 2d csc matrix or a 2d dense array
            if use_sketch and A.shape[0] > 5 * out_dim[i]:
                # there is benefit to sketch
                A = cs.gen_apply(A, out_dim[i])
                
            if not isinstance(A, T.lib.ndarray):
                # A is SpTensor
                A = A.toarray()
            
            #! after sketch, call matrix ID on A
            J, X = solver.select(A, ranks[i], rule = selection_rule)
            T = T.take(J, i) 
                                
            idx_sets[i] = J
            satellites[i] = X  # of shape (rank, n)
            _, Rs[i] = self.lib.linalg.qr(X.T)
            
        return idx_sets, satellites  # in the order of axis
    
                
