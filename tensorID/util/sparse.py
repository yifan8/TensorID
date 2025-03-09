# gpu utils for sparse tensor

import numpy as np
from collections.abc import Iterable
from .KFJLT import KFJLT
from .hadamard import hadamard
from .misc import next_pow2, numel



letters = 'abcdefghijklmnopqrstuvwxyz'
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

class SpTensor:
    
    def __init__(self, idx, val, shape, device = 'auto'):
        if device == 'cpu':
            self.lib = np
            import scipy.sparse as sps
            self.sps = sps
        else:
            gpu_available = True
            try:
                import cupy as cp
                import cupyx.scipy.sparse as cpsps
            except:
                gpu_available = False
            if device == 'auto':
                self.lib = cp if gpu_available else np
                self.sps = cpsps if gpu_available else sps
            elif device == 'gpu':
                if gpu_available:
                    self.lib = cp
                    self.sps = cpsps
                else:
                    raise ValueError('gpu library is not available')
            else:
                raise ValueError(f'device = {device} is not understood')
        
        self.gpu = (self.lib != np)
        self.device = 'cpu' if not self.gpu else 'gpu'
        
        self.idx = self.lib.array(idx)
        self.val = self.lib.array(val)
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
    
    
    def __imul__(self, other):
        # for now only take other = float
        self.val *= other
        return self
    
    
    def __mul__(self, other):
        return SpTensor(self.idx, self.val * other, self.device)
    
    
    def norm(self, order = 2):
        return self.lib.linalg.norm(self.val, ord = order)
        
    
    def copy(self, device = None):
        if device is None:
            device = self.device
        return SpTensor(self.idx.copy(), self.val.copy(), self.shape, device = device)
    
    
    def from_sparse_matrix(self, M):
        row, col, val = self.sps.find(M)
        M = SpTensor(self.lib.array([row, col]).T, val, shape = M.shape, device = self.device)
        return M
    
    
    def from_array(self, M):
        shape = np.array(M.shape, dtype = np.int64)
        M = self.lib.array(M).reshape(-1)
        idx = self.lib.flatnonzero(M)
        val = M.ravel()[idx]
        idx = self.lib.unravel_index(idx, shape)
        idx = self.lib.array(idx).T
        return SpTensor(idx, val, shape = tuple(shape), device = self.device)
        
    
    
    def to_sparse_matrix(self, spmat):
        if self.ndim != 2:
            raise ValueError(f'current array has dimension {self.ndim}, cannot be converted')
        return spmat((self.val, (self.idx[:, 0], self.idx[:, 1])), shape = self.shape)
    
    
    def get_idx(self):
        return self.idx.get() if self.gpu else self.idx
    
    
    def get_val(self):
        return self.val.get() if self.gpu else self.val
    
    
    def get_shape(self):
        return self.shape
    
    
    def set_shape(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
    
    
    def reshape(self, *shape, order = 'C'):
        if isinstance(shape[0], Iterable):
            # order is iterable
            shape = np.array(shape[0], dtype = np.int64)
        else:
            shape = np.array(shape, dtype = np.int64)
        oldshape = self.shape
        size = numel(oldshape)
        newsize = numel(shape)
        if -1 in shape:
            val = int((size + 0.5 * (-newsize)) // (-newsize))
            shape[shape == -1] = val
        newsize = numel(shape)
        if newsize != size:
            raise ValueError(f'target shape {tuple(shape)} is not compatible with input shape {tuple(oldshape)}')
        
        if tuple(oldshape) == tuple(shape):
            # no reshaping is needed
            Warning(f'input tensor already has shape {tuple(shape)}, no operation was done, the input was returned.')
            return self
        
        idx = self.lib.ravel_multi_index(self.idx.T, np.array(oldshape, dtype = np.int64), order = order)
        idx = self.lib.unravel_index(idx, tuple(shape), order = order)
        idx = self.lib.array(idx).T
        return SpTensor(idx, self.val, shape, device = self.device)
    
    
    def prune(self, axis, return_map = False):
        # remove empty slices i for which T[..., i, ...] = 0
        # require axis to be an integer
        # if want to remove slices indexed by 2 args, should reshape first
        # if return_map is True, then return a list 
        # list[i] = (idx in the original tensor of the ith slice after the prune)
        if axis < -self.ndim or axis > self.ndim-1:
            raise ValueError(f'axis = {axis} is out of bound, dimension is {self.ndim}')
        if axis < 0:
            axis = axis + self.ndim
        idx = self.idx[:, axis]
        idx = idx[(order := self.lib.argsort(idx))]  # from min to max
        self.idx[order, axis] = self.lib.cumsum(self.lib.diff(idx, prepend = idx[0]) > 0)
        
        shape = list(self.shape)
        shape[axis] = int(self.idx[:, axis].max()) + 1
        self.set_shape(tuple(shape))
        
        if return_map:
            return self.lib.unique(idx)
    
    
    def unprune(self, axis, idxmap, newlen = None):
        if axis < -self.ndim or axis > self.ndim-1:
            raise ValueError(f'axis = {axis} is out of bound, dimension is {self.ndim}')
        if axis < 0:
            axis = axis + self.ndim
        
        self.idx[:, axis] = idxmap[self.idx[:, axis]]
        shape = np.array(self.shape)
        if newlen is None:
            shape[axis] = idxmap[-1] + 1
        else:
            shape[axis] = newlen
        self.set_shape(shape)
        
    
    
    def vec(self, order = 'C'):
        # returns a csc formatted (size, 1) vectorization
        idx = self.lib.ravel_multi_index(self.idx.T, np.array(self.shape, dtype = np.int64), order = order)
        return self.sps.csc_matrix(
            (self.val, (idx, self.lib.zeros(len(self.val)))),
            shape = (self.size(), 1)
            )
        

    def transpose(self, *order, axis = None):
        if len(order) == 1:
            # order is iterable
            order = np.array(order[0])
        else:
            order = np.array(order)
        order = self._convert_axis_input(order, self.ndim)
        if axis is not None:
            idx = self.idx.copy()
            idx[:, axis] = self.idx[:, axis[order]]
            shape = np.array(self.shape)
            shape[:, axis] = shape[:, axis[order]]
        else:
            idx = self.idx[:, order]
            shape = np.array(self.shape)[order]
        return SpTensor(idx, self.val, shape, device = self.device)
            
    
    
    def take(self, idx, axis):
        coord = self.idx[:, axis]
        sel = self.lib.isin(coord, self.lib.array(idx))
        invmap = {int(idx[i]) : -i-1 for i in range(len(idx))}
        subs = self.idx[sel]
        for j in idx:
            j = int(j)
            row = subs[:, axis] == j
            subs[row, axis] = invmap[j]
        subs[:, axis] = -subs[:, axis] - 1
        vals = self.val[sel]
        shape = np.array(self.shape)
        shape[axis] = len(idx)
        return SpTensor(subs, vals, shape, device = self.device)
    
    
    def todense(self):
        res = self.lib.zeros(self.shape)
        res[*self.idx.T] = self.val
        return res 
    
    
    def get_dense(self):
        # return densified array to cpu
        return self.todense().get() if self.gpu else self.todense()
    
    
    def nnz(self):
        return len(self.idx)
    
    
    def size(self):
        return numel(self.shape)
    
    
    def density(self):
        return self.nnz() / self.size()
    
    
    def partition_axes(self, I, J = None, inplace = True):
        # partition axes to T(I, J), NOT reshaping to matrix
        # if I or J is not specified, use natural ordering
        # this is by default an INPLACE operation
        permu, _ = self._get_matricization_data(self, I, J)
        if inplace:
            self.idx = self.idx[:, permu]
            return None
        else:
            return self.transpose(permu)
    
    
    def mat(self, I, J = None, order = 'C'):
        # T.mat(I, J) returns mat(I, J)T
        # J is always I complement, if J is skipped, then order axes in J in natural order
        # if I = -1, and J is specified then mat(J^c, J) is returned, order I in natural order
        permu, nrow = self._get_matricization_data(self, I, J)
        res = self.transpose(*permu)
        res = res.reshape(nrow, -1, order = order)
        return res
    
    
    @classmethod
    def _get_matricization_data(cls, tns, I, J = None):
        dim = len(tns.shape)
        shape = np.array(tns.shape, dtype = np.int64)
        size = numel(shape)
        
        I = cls._convert_axis_input(I, dim)
        J = cls._convert_axis_input(J, dim)
            
        if J is None:
            # I, None
            J = list(set([i for i in range(dim)]).difference(set(I)))
            permu = np.zeros(dim, dtype = np.int64)
            permu[:len(I)] = I
            permu[len(I):] = np.sort(J)
            nrow = numel(shape[I])
        elif I is None:
            # None, J
            I = list(set([i for i in range(dim)]).difference(set(J)))
            permu = np.zeros(dim, dtype = np.int64)
            permu[:len(I)] = np.sort(I)
            permu[len(I):] = J
            nrow = int(size / numel(shape[J]))
        else:
            # I, J
            permu = np.array(list(I) + list(J), dtype = np.int64)
            nrow = numel(shape[I])
        return permu, nrow
    
    
    def dotv(self, v, axis = None, density_thres = 0.2):
        # compute T(..., i) \times_(-1) v[i]
        # if ndim(v) > 1, v will be reshaped into vector
        # if axis is not None, then -1 above is replaced by axis, if axis is iterable, this flattens T over axis
        # this will then compute T(..., i, j, k, l, ...) v[k]
        # return shape is Tv(..., i, j, l, ...)
        # will densify the array and return ndarray if density > density_thres
        
        axis = self._convert_axis_input(axis, self.ndim)
        if axis is None:
            axis = [self.ndim - 1]
        
        if v.ndim > 1:
            v = v.reshape(-1)
        outshape = [self.shape[i] for i in range(self.ndim) if i not in axis]
        out = self.reshape(*outshape, -1)
        out = out.ttv([v], axis = -1, keep_sparse = True)
        if out.density() > density_thres:
            return out.todense()
        return out
    
    
    def ttv(self, vecs, axis = None, fold_back = True, keep_sparse = False):
        # contract T with vecs[k] on mode k for all k in axis
        # the contracted mode is removed from tensor
        # if axis is not None, then k is axis[k], if None, then k = 0,...,d-1
        # will first do the ttv that reduces the size the most
        # will return a dense array
        
        nvec = len(vecs)
        axis = self._convert_axis_input(axis, self.ndim)
        if axis is None:
            axis = [i for i in range(nvec)]
        out = self.copy()
        for i in range(nvec):
            idx = self.idx[:, axis[i]]
            out.val *= vecs[i][idx]
        out = out.mat(None, axis)
        out.prune(axis = 1)
        out = out.to_sparse_matrix(self.sps.csr_matrix)
        outshape = tuple([self.shape[i] for i in range(self.ndim) if i not in axis])
        if len(outshape) == 0:
            # this is an inner product reducing to a scalar
            return out.sum(axis = 1).item()
        
        if keep_sparse:
            idx = self.lib.flatnonzero(self.lib.diff(out.indptr))
            val = self.lib.array(out[idx].sum(axis = 1)).reshape(-1)
            
            if not fold_back:
                return self.sps.csc_matrix((val, (idx, self.lib.zeros(len(idx)))), shape = (out.shape[0], 1))
            
            idx = self.lib.array(self.lib.unravel_index(idx, outshape)).T
            return SpTensor(idx, val, shape = outshape, device = self.device)
        
        # use dense
        out = self.lib.array(out.sum(axis = 1))
        if fold_back:
            out = out.reshape(outshape)
        return out
    
    
    def ttd(self, diags, axis = None, inplace = False):
        # compute tensor times diagonal matrices
        # diags is a list of diagonal values
        nmat = len(diags)
        if nmat > self.ndim:
            raise ValueError(f'number of matrices {nmat} is greater than number of modes {self.ndim}')
        if nmat == 0:
            return self
        axis = self._convert_axis_input(axis, self.ndim)
        if axis is None:
            axis = [i for i in range(len(diags))]
        
        if inplace:
            M = self
        else:
            M = self.copy()
        for i in range(nmat):
            idx = M.idx[:, axis[i]]
            M.val *= M.lib.array(diags[i][idx])
        if not inplace:
            return M
        return None
        
    
    def ttm(self, mats, axis = None, density_thres = 0.2, suppress_singleton_dim = True):
        # compute T(:, i, :) \times_(k) mats[k](i, j)
        # if axis is not None, then k is axis[k]
        # will first do the ttm that reduces the size the most
        # will densify the array and return ndarray if density > density_thres
        # if suppress_singleton_dim, the mode of length 1 will be removed in the result (this happens when one of the mats are matrices of shape (x, 1))
        
        nmat = len(mats)
        if nmat > self.ndim:
            raise ValueError(f'number of matrices {nmat} is greater than number of modes {self.ndim}')
        if nmat == 0:
            return self
        axis = self._convert_axis_input(axis, self.ndim)
        if axis is None:
            axis = [i for i in range(len(mats))]
        
        outdims = np.array([mats[i].shape[1] / self.shape[axis[i]] for i in range(len(mats))])
        order = np.argsort(outdims)
        
        n = len(mats)
        M = self
        for i in range(n):
            k = order[i].item()
            mat = mats[k]
            dim = axis[k] if axis is not None else k
            shape_after_ttm = np.array(M.shape)
            shape_after_ttm[dim:-1] = M.shape[dim+1:]
            if suppress_singleton_dim and mat.shape[1] == 1:
                shape_after_ttm = shape_after_ttm[:-1]
            else:
                shape_after_ttm[-1] = mat.shape[1]

            skip_transpose = (dim == M.ndim-1) or (suppress_singleton_dim and mat.shape[1] == 1)
            if not skip_transpose:
                neworder = [i for i in range(dim)] + [self.ndim-1] + [i for i in range(dim, self.ndim-1)]
            
            if isinstance(M, SpTensor):
                M = M.mat(None, dim)  # (i1,...,i(k-1), i(d+1),...,id)
                nrow = M.shape[0]
                rowidx = M.prune(axis = 0, return_map = True)
                M = M.to_sparse_matrix(self.sps.csc_matrix)
                M = M @ mat
                M = self.from_array(M)
                M.unprune(0, rowidx, nrow)
            else:
                # M is np.array or cp.array
                permu, nrow = self._get_matricization_data(M, None, dim)
                M = M.transpose(permu)
                M = M.reshape(nrow, -1)
                M = M @ mat
            if isinstance(M, self.sps.spmatrix):
                density = M.nnz / numel(M.shape)
                if density > density_thres:
                    # convert M to dense
                    M = M.toarray()
            if len(shape_after_ttm) > 0:
                M = M.reshape(*shape_after_ttm)
            if not skip_transpose:
                M = M.transpose(*neworder)  # (i1,...,i(d-1), jd, i(d+1),...)
        
        return M
    
    
    def mttkrp(self, mats, axis = None, fold_back = True, density_thres = 0.2):
        # compute T(a, ..., i, j, ..., z) \times_(0) mats[0](i, r) \times_(1) mats[1](j, r) -> T(a, ..., z, r)
        # mats has shape (#axis, mode_len, rank)
        # if axis is not None, then \times_(axis[i]) mats[i] ...
        # will first do the ttm that reduces the size the most
        # will return a dense ndarray
        
        nmat = len(mats)
        if nmat == 0:
            return self
        if nmat > self.ndim:
            raise ValueError(f'number of matrices {nmat} is greater than number of modes {self.ndim}')
        axis = self._convert_axis_input(axis, self.ndim)
        if axis is None:
            axis = [i for i in range(nmat)]
        
        R = mats[0].shape[1]
        outshape = [self.shape[i] for i in range(self.ndim) if i not in axis] + [R]
        outdim = numel(outshape[:-1])

        out = self.sps.csc_matrix((outdim, R))
        for r in range(R):
            outr = self.ttv([mat[:, r] for mat in mats], axis, keep_sparse = True, fold_back = False)
            out[:, r] = outr 
        density = out.nnz / numel(out.shape)
        if density > density_thres:
            out = out.toarray()
            if not fold_back:
                return out
            else:
                return out.reshape(outshape)
        
        # keep sparse, return SpTensor
        out = self.from_sparse_matrix(out)
        out = out.reshape(outshape)
        return out
    
    
    def _kfjlt(self, row_idx, signs, axis = None, density_thres = 0.2):
        # apply KFJLT to axis given row idx and signs
        # row_idx is (#factors, n_sample)
        # signs is a list of sign vectors, has length #factors
        # length of signs[i] must be a power of 2 and greater than self.shape[i], or [axis[i]] if axis is not None
        # will put the KFJLT mode at front, and the rest in netural order
        
        nfac = len(signs)
        if nfac > self.ndim:
            raise ValueError(f'KFJLT order {nfac} is greater than number of modes {self.ndim}')
        axis = self._convert_axis_input(axis, self.ndim)
        if axis is None:
            axis = [i for i in range(nfac)]
        
        n = len(row_idx[0])
        kf_shape = list(self.shape)
        bin_shape = []  # reshape tensor to apply mttkrp
        # we use 128 per each dim
        default_size = 128 
        log_default_size = 7
        kron_orders = []  # this is log2(bin_shape), how many H matrices to apply
        for i in range(nfac):
            li = len(signs[i])
            if li <= default_size:
                bin_shape.append(li)
                kron_orders.append(np.round(np.log2(li)))
            else:
                logli = np.round(np.log2(li))
                n_default = int(logli // log_default_size)
                n_rem = int(logli % log_default_size)
                bin_shape += [default_size] * n_default
                kron_orders += [log_default_size] * n_default
                if n_rem > 0:
                    bin_shape.append(int(2**n_rem))
                    kron_orders.append(n_rem)
            kf_shape[axis[i]] = li
        M = self.copy()
        M.set_shape(kf_shape)
        M.partition_axes(axis, None)
        kf_shape = np.array(M.shape, dtype = np.int64)
        
        # apply signs
        M = M.ttd(signs)
        
        # apply hadamard
        bin_dim = len(bin_shape)
        M = M.reshape(bin_shape + list(kf_shape[nfac:]))
        
        # get ttv vectors
        row_idx_bin = self.lib.ravel_multi_index(row_idx, kf_shape)
        row_idx_bin = self.lib.array(self.lib.unravel_index(row_idx_bin, bin_shape))
        # row_idx_bin has shape (#bin dim, n)
        
        vecs = []
        H0 = self.lib.array([[1, 1], [1, -1]]) / self.lib.sqrt(2)
        Hs = [H0]
        for j in range(1, log_default_size):
            Hs.append(self.lib.kron(Hs[-1], H0))
        for j in range(bin_dim):
            vecs.append(Hs[kron_orders[j] - 1][:, row_idx_bin[j]])  # 2**kron_orders[j] by n
        
        M = M.mttkrp(vecs, fold_back = True, density_thres = density_thres)  # (natural order, KF_dim) SpTensor or array
        if isinstance(M, SpTensor):
            M.partition_axes(-1, None)
        else:
            # M is ndarray
            M = M.transpose([M.ndim-1] + [i for i in range(M.ndim-1)])
        return M * self.lib.sqrt(numel(bin_shape) / n)
    
    
    def kfjlt(self, axis, sketch_dim, seed = None, density_thres = 0.2):
        # generate a random KFJLT operator and apply to T
        axis = self._convert_axis_input(axis, self.ndim)
        nfac = len(axis)
        in_shapes = [next_pow2(axis[j]) for j in range(nfac)]
        if seed is not None:
            self.lib.random.seed(seed)
        row_idx, signs = KFJLT(self.device).get_operator(sketch_dim, in_shapes)
        return self._kfjlt(row_idx, signs, axis, density_thres)
        
        
        
    def rand_dist_to_tucker(self, core, factors, sketch_dim, seed = None):
        # evaluate the distance to Tucker(core, factors)
        # self has shape IJK, core is ijk and factors are Ii, Jj, Kk...
        # usually factors are very tall and thin
        # uses KFJLT to approx the dist
        # core can be a sparse tensor
        
        kf = KFJLT(backend = self.device, seed = seed)
        
        dense_core = not isinstance(core, SpTensor)
        if dense_core:
            core = self.lib.array(core)
            
        nfac = len(factors)
        ranks = self.lib.zeros(nfac, dtype = self.lib.int64)
        kf_shape = list(self.shape)
        for i in range(self.ndim):
            kf_shape[i] = next_pow2(kf_shape[i])
            
        row_idx, signs = kf.get_operator(sketch_dim, kf_shape)
        for i in range(nfac):
            factors[i] = self.lib.array(factors[i])
            ranks[i] = factors[i].shape[1]
            signs[i] = self.lib.array(signs[i]).reshape(-1)
        
        v1 = self._kfjlt(row_idx, signs, density_thres = 0)  # 1d-array
        
        transformed_factors = []
        for dim in range(nfac):
            mat = self.lib.zeros((kf_shape[dim], ranks[dim].item()))
            mat[:self.shape[dim]] = signs[dim][:self.shape[dim], None] * factors[dim]
            transformed_factors.append(hadamard(mat)[row_idx[dim]].T)  # sketch_dim, shape[i]
        
        if dense_core:
            signature = ', '.join([letters[i] + 'Z' for i in range(nfac)] + [letters[:nfac]]) + '-> Z'
            v2 = self.lib.einsum(signature, *transformed_factors, core, optimize = 'optimal')
        
        else:
            v2 = core.mttkrp(transformed_factors, density_thres = 0, fold_back = False)  # (1, sketch_dim) array
            v2 = (v2 * np.sqrt(numel(kf_shape) / sketch_dim)).reshape(-1)
        
        return self.lib.linalg.norm(v1 - v2)
            
            
    @staticmethod
    def _convert_axis_input(axis, dim):
        if axis is None:
            return None
        if isinstance(axis, int): 
            if axis < 0:
                axis = dim + axis
                if axis < 0:
                    raise ValueError(f'index {axis} out of bound (0 to {dim-1}) after conversion from negative')
            axis = [axis]
        elif isinstance(axis, Iterable):
            for i in range(len(axis)):
                if axis[i] < 0:
                    axis[i] += dim
                    if axis < 0:
                        raise ValueError(f'index {axis} out of bound (0 to {dim-1}) after conversion from negative')
                if axis[i] >= dim:
                    raise ValueError(f'index {axis} out of bound (0 to {dim-1})')
        return axis
            
        
            
        
    