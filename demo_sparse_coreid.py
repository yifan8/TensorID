import numpy as np

from tensorID.coreid import SparseSolver
from tensorID.util import SpTensor

import warnings
warnings.filterwarnings(action = 'ignore')

from time import perf_counter as now

class Arg:
    kf_dim = 400
    cs_dim = 1000
    out_dim = 1000
    sketch_dim = (kf_dim, cs_dim, out_dim)  
    order = [2, 0, 1]
    
    backend = 'cpu'
    datafile = f'data/nell_large.npz'
    selection_rule = 'inf'
    nrun = 3
    seed = 1234
    r0 = 50
    r1 = 200
    rstep = 50
    ranks = np.arange(r0, r1 + 0.1, rstep, dtype = np.int64)
    
args = Arg()


letters = 'abcdefghijklmnopqrstuvwxyz'
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# read in the data
data = np.load(args.datafile)
key = data['key']
value = data['value']
shape = tuple(data['shape'])

T = SpTensor(key, value, shape, device = args.backend)
T_norm = np.linalg.norm(value)

d = len(T.shape)
for j in range(d):
    T.prune(axis = j)
    
m = args.sketch_dim

# start the simulation
solver = SparseSolver()
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
        idx, sate = solver.fit(
            T, 
            [rank for _ in range(d)],
            axis = args.order,
            sketch_dim = args.sketch_dim,
            selection_rule = args.selection_rule,
            seed = seeds[i]
        )
        t1 = now()
        print(f'\t\tCoreID done, time = {t1 - t0:.3e} seconds, ', end = '')
        
        
        # compute reconstruction error
        core = T.take(idx[0], args.order[0])
        for dim in range(1, d):
            core = core.take(idx[dim], args.order[dim])
        
        temp = np.argsort(args.order)
        factors = [None for _ in range(d)]
        for dim in range(d):
            factors[dim] = sate[temp[dim]].T

        # use randomized methods to estimate reconstruction error
        err = T.rand_dist_to_tucker(core, factors, sketch_dim = 200)
        err = err / T_norm
        err_cid[i, j] = err
        time_cid[i, j] = t1 - t0
        print(f'err = {err:.6f}')


# visualize
import matplotlib.pyplot as plt
err_cid_mean = np.mean(err_cid, axis = 0)
plt.figure(figsize = (8, 8))
plt.plot(args.ranks, err_cid_mean, 'r--x', label = 'CoreID')
plt.legend(loc = 'best')
plt.grid(True)
plt.show()






