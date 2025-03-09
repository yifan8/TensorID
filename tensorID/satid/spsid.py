from ..util import SpTensor

letters = 'abcdefghijklmnopqrstuvwxyz'
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

class SparseSolver:
    
    def __init__(self, backend: str = 'cpu'):
        """SatID solver for sparse tensor

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
            self.la = numpy.linalg
        else:
            try:
                import cupy
            except:
                raise ValueError(f'got backend = {backend} but gpu library cupy not available')
            self.lib = cupy
            from cupyx.scipy.sparse import linalg as cspsla
            self.la = cspsla
    
    
    def fit(self, tns: SpTensor, ranks: list[int], 
            axis: list[int] = None, 
            selection_rule: int = 'inf',
            seed: None | int = None) -> tuple[list, list]:
        """Compute the SatID

        Parameters
        ----------
        T : SpTensor
            the sparse tensor to be decomposed
        ranks : list[int]
            target rank of SatID   
        axis : list[int], optional
            axis from which the indices are selected, e.g., [2, 0] for a 3rd order tensor means first select from axis 2 then axis 0, no selection is made on axis 1, if None, this defaults to [0, ..., d-1], by default None
        selection_rule : str | int, optional
            selection rule, 'inf' or a positive number f, for which a sampling is made according to pmf = column norms to power f, by default 'inf'
        seed : None | int, optional
            random seed, by default None

        Returns
        -------
        tuple[list, list]
            1. index sets on each mode,
            2. satellite matrices on each mode
        """
        
        if seed is not None:
            self.lib.random.seed(seed)
            
        nr = len(ranks)
        if axis is None:
            axis = [i for i in range(nr)]
        
        d = tns.ndim
        idx_sets = []
        satellites = []
            
        for dim in range(nr):
            # reorder axis as [ax_sel, ax1, ax2, ...]
            A = tns.mat(axis[dim], None)  # tns.shape[dim] * N
            idxmap = A.prune(axis = 1, return_map = True)
            A = A.to_sparse_matrix(A.sps.csc_matrix)
            n, N = A.shape
            k = ranks[dim]  # rank[i] is the rank on mode axis[i]
            idx_shape = tns.lib.array([tns.shape[i] for i in range(d) if i != dim], dtype = self.lib.int64)

            # select cols of A
            if selection_rule == 0:
                # uniform
                J = self.lib.random.choice(self.lib.arange(N), replace = False, size = k)
                idx = idxmap[J]
                fac = tns.from_sparse_matrix(A[:, J])  # n * rank
                # undo the prune
                fac.idx[:, 0] = idxmap[fac.idx[:, 0]]
                fac.set_shape((tns.shape[dim], k))
                
                idx_sets.append(idx)
                satellites.append(fac)
            
            else:
                J = []
                score = tns.lib.array(A.power(2).sum(axis = 0)).reshape(-1)
                Q = tns.lib.zeros((n, k))
                for i in range(k):
                    if selection_rule == 'inf':
                        # greedy maximal score
                        idx = int(tns.lib.argmax(score))
                    elif isinstance(selection_rule, int | float):
                        rem = self.lib.argwhere(score > 0).reshape(-1)
                        p = (temp := score[rem]**selection_rule) / self.lib.sum(temp)
                        idx = int(tns.lib.random.choice(rem, size = 1, p = p))
                    else:
                        raise ValueError(f'selection rule {selection_rule} not understood')
                    J.append(idx)
                    
                    # update Q, R and solve for the LS weights w
                    sel_col = tns.lib.array(A[:, idx].todense()).reshape(-1)
                    if i > 0:
                        q = sel_col - Q[:, :i] @ (Q[:, :i].T @ sel_col) 
                        q /= self.lib.linalg.norm(q)
                    else:
                        q = sel_col / tns.lib.linalg.norm(sel_col)
                    Q[:, i] = q
                    
                    # update scores for next selection
                    score -= (A.T @ q)**2 
                    score[score < 1E-10] = 0
                
                fac = tns.from_sparse_matrix(A[:, J])  # n, rank
                # undo the prune
                fac.set_shape((tns.shape[dim], k))
                idx = tns.lib.unravel_index(idxmap[J], idx_shape)
                satellites.append(fac)
                idx_sets.append(tns.lib.array(idx))
                
                # end of the selection of ith column
            # end of the selection over mode dim
        # complete the selection over all modes 
        return idx_sets, satellites

                    
                    
    def solve_core(self, tns: SpTensor, 
                   satellites: list[SpTensor], 
                   axis: list[int] = None, 
                   sigma_thres: float = 1E-6, 
                   compute_error: bool = False, 
                   compute_core: bool = True):
        """Compute the core tensor

        Parameters
        ----------
        tns : SpTensor
            the tensor on which SatID is done
        satellites : list[SpTensor]
            list of satellite nodes solved by SatID
        axis : list[int], optional
            list of axis, see SparseSolver.fit, by default None
        sigma_thres : float, optional
            regularization threshold used in least squares solve, singular values of the coefficient matrix below sigma_thres is dropped, by default 1E-6
        compute_error : bool, optional
            whether the reconstruction error is computed and returned, by default False
        compute_core : bool, optional
            whether the core tensor is computed and returned, if only the error is needed, this should be set to False to save computation, by default True

        Returns
        -------
        reconstruction error and/or the Tucker core of SatID
        """
        
        Us = []  # n * rank
        VSinvs = []
        d = tns.ndim
        
        if axis is None:
            axis = [i for i in range(d)]
        n_axis = len(axis)

        if compute_error:
            tns_norm = tns.norm()
            
        for sate in satellites:
            # need to compute sate.T.dagger @ tensor
            sate = sate.todense()  # n, rank
            U, s, VT = tns.lib.linalg.svd(sate, full_matrices = False)
            sel = s > sigma_thres
            Us.append(U[:, sel])
            VSinvs.append(VT[sel].T * s[sel]**(-1))
        
        core = tns.ttm(Us, axis = axis)
        
        if compute_error:
            if isinstance(core, SpTensor):
                proj_norm = core.norm()
            else:
                proj_norm = tns.lib.linalg.norm(core)
            err = (tns_norm**2 - proj_norm**2)**(1/2)
        
        if compute_core:
            if isinstance(core, SpTensor):
                core = core.todense()
            # e.g. aA, dD, cC, ABCD -> abcd for axis = [0, 3, 2]
            out_str = ''
            for i in range(d):
                if i in axis:
                    out_str += letters[i]
                else:
                    out_str += LETTERS[i]
            signature = ','.join([letters[axis[i]] + LETTERS[axis[i]] for i in range(n_axis)] + [LETTERS[:d]]) + '->' + out_str
            core = tns.lib.einsum(signature, *VSinvs, core, optimize = 'optimal')
        
        if compute_error:
            if compute_core:
                return core, err
            return err
        return core
                        