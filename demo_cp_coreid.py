import numpy as np

from tensorID.coreid import CPSolver
from tensorly.decomposition import tucker

import warnings
warnings.filterwarnings(action = 'ignore')

from time import perf_counter as now


class Arg:
    sketch_dim = 128
    backend = 'cpu'
    datafile = f'data/cp128_4.npz'
    selection_rule = 2
    nrun = 3
    seed = 1234
    r0 = 4
    r1 = 12
    rstep = 4
    tucker = False
    ranks = np.arange(r0, r1 + 0.1, rstep, dtype = np.int64)
    
args = Arg()



letters = 'abcdefghijklmnopqrstuvwxyz'
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# read in the data
data = np.load(args.datafile)
factors = data['factors']

if args.backend == 'gpu':
    import cupy as cp
    factors_gpu = []
    for i in range(len(factors)):
        factors_gpu.append(cp.asarray(factors[i]))

T_norm = data['T_norm']

d = len(factors)
m = args.sketch_dim
sketch_dim = [m for _ in range(d)] if m is not None else None

signature = ', '.join([letters[i] + 'X' for i in range(d)]) + '->' + letters[:d]
T = np.einsum(signature, *factors, optimize = 'optimal')


# start the simulation
solver = CPSolver(backend = args.backend)
err_cid = np.zeros((args.nrun, len(args.ranks)))
err_tk = np.zeros(len(args.ranks))
time_cid = np.zeros((args.nrun, len(args.ranks)))
time_tucker = np.zeros(len(args.ranks))

if args.seed is not None:
    np.random.seed(args.seed)

seeds = np.random.randint(100000000, size = args.nrun)
for i in range(args.nrun):
    print(f'test {i+1}/{args.nrun}')
    for (j, rank) in enumerate(args.ranks):
        print(f'\trank = {rank}')
        t0 = now()
        idx_sets, satellites, _ = solver.fit(
            factors if args.backend == 'cpu' else factors_gpu, 
            [rank for _ in range(d)],
            sketch_dim = sketch_dim,
            selection_rule = args.selection_rule,
            seed = seeds[i]
        )
        t1 = now()
        
        if args.backend == 'gpu':
            for k in range(len(satellites)):
                satellites[k] = satellites[k].get()
        
        selected_factors = [factors[i][idx_sets[i]] for i in range(d)]
        signature = ', '.join([letters[i] + 'X' for i in range(d)]) + '->' + letters[:d]
        core = np.einsum(signature, *selected_factors, optimize = 'optimal')
        signature = ', '.join([LETTERS[i] + letters[i] for i in range(d)] + [LETTERS[:d]]) + '->' + letters[:d]
        recon = np.einsum(signature, *satellites, core, optimize = 'optimal')
        err_cid[i, j] = (err := np.linalg.norm(recon - T) / T_norm)
        time_cid[i, j] = t1 - t0
        print(f'\t\tCoreID done, err = {err:.3f}, time = {t1 - t0:.3e} seconds')
        
        if i == 0 and args.tucker:
            t0 = now()
            opt_core, opt_factors = tucker(T, [rank for _ in range(d)], n_iter_max = 30, init = 'random')
            t1 = now()
            
            # optimal reconstruction error
            signature = ', '.join([letters[i] + LETTERS[i] for i in range(d)] + [LETTERS[:d]]) + '->' + letters[:d]
            opt_T = np.einsum(signature, *opt_factors, opt_core, optimize = 'optimal')
            err_tk[j] = (err := np.linalg.norm(opt_T - T) / T_norm)
            time_tucker[j] = t1 - t0
            print(f'\t\tTucker done, err = {err:.3f}, time = {t1 - t0:.3e} seconds')
    


# visualize
import matplotlib.pyplot as plt
err_cid_mean = np.mean(err_cid, axis = 0)
plt.figure(figsize = (8, 8))
if args.tucker:
    plt.plot(args.ranks, err_tk, 'k--x', label = 'Tucker')
plt.plot(args.ranks, err_cid_mean, 'r--x', label = 'CoreID')
plt.legend(loc = 'best')
plt.grid(True)
plt.show()






