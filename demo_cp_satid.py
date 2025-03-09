import numpy as np

from tensorID.satid import CPSolver
from tensorly.decomposition import tucker

import warnings
warnings.filterwarnings(action = 'ignore')

from time import perf_counter as now


letters = 'abcdefghijklmnopqrstuvwxyz'
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


seed = 1234
fname = f'data/cp128_4.npz'
nrun = 3
sketch_dim = 256
run_tucker = False
r0 = 4
r1 = 12
rstep = 4
backend = 'cpu'

#! import data
data = np.load(fname)
factors = data['factors']
T_norm = data['T_norm']

if backend == 'gpu':
    import cupy as cp
    factors_gpu = []
    for i in range(len(factors)):
        factors_gpu.append(cp.asarray(factors[i]))

d = len(factors)
solver = CPSolver(backend = backend)

#! start the simulation
ranks = np.arange(r0, r1 + 0.1, rstep, dtype = np.int64)  # target ranks
err_sid = np.zeros((nrun, len(ranks)))
err_tk = np.zeros(len(ranks))
time_sid = np.zeros((nrun, len(ranks)))
time_tk = np.zeros(len(ranks))

signature = ','.join([letters[i] + 'X' for i in range(d)]) + '->' + letters[:d]
T = np.einsum(signature, *factors, optimize = 'optimal')

np.random.seed(seed)
seeds = np.random.randint(10000000, size = nrun * len(ranks))
for k in range(nrun):
    print(f'test {k+1}/{nrun}:')
    for (j, rank) in enumerate(ranks):
        print(f'\trank = {rank}')
        t0 = now()
        idx_sets, sates, new_factors = solver.fit(
            factors if backend == 'cpu' else factors_gpu, 
            [rank for _ in range(d)],
            sketch_dim = sketch_dim,
            seed = seeds[k * len(ranks) + j]
        )
        t1 = now()
        
        signature = ','.join([letters[i] + 'X' for i in range(d)]) + '->' + letters[:d]
        if backend == 'gpu':
            for i in range(len(new_factors)):
                new_factors[i] = new_factors[i].get()
        recon = np.einsum(signature, *new_factors, optimize = 'optimal')
        err_sid[k, j] = np.linalg.norm(recon - T) / T_norm
        time_sid[k, j] = t1 - t0
        print(f'\t\tSatID done, err = {err_sid[k, j]:.3f}, time = {t1 - t0:.3e} seconds')
        
        if k == 0 and run_tucker:
            t0 = now()
            opt_core, opt_factors = tucker(T, [rank for _ in range(d)], n_iter_max = 100, init = 'random')
            t1 = now()
            
            # optimal reconstruction error
            tucker_signature = ','.join([letters[i] + LETTERS[i] for i in range(d)] + [LETTERS[:d]]) + '->' + letters[:d]
            opt_T = np.einsum(tucker_signature, *opt_factors, opt_core, optimize = 'optimal')
            err_tk[j] =np.linalg.norm(opt_T - T) / T_norm
            time_tk[j] = t1 - t0
            print(f'\t\tTucker done, err = {err_tk[j]:.3f}, time = {t1 - t0:.3e} seconds')

# visualize
import matplotlib.pyplot as plt
err_cid_mean = np.mean(err_sid, axis = 0)
plt.figure(figsize = (8, 8))
if run_tucker:
    plt.plot(ranks, err_tk, 'k--x', label = 'Tucker')
plt.plot(ranks, err_cid_mean, 'r--x', label = 'CoreID')
plt.legend(loc = 'best')
plt.grid(True)
plt.show()






