import numpy as np

from tensorID.satid import SparseSolver
from tensorID.util import SpTensor

import warnings
warnings.filterwarnings(action = 'ignore')



class Arg:
    backend = 'gpu'
    datafile = f'data/enron_large.npz'
    selection_rule = 'inf'
    nrun = 3
    seed = 1234
    r0 = 50
    r1 = 200
    rstep = 50
    ranks = np.arange(r0, r1 + 0.1, rstep, dtype = np.int64)
    
args = Arg()

# read in the data
data = np.load(args.datafile)
key = data['key']
value = data['value']
shape = tuple(data['shape'])

T = SpTensor(key, value, shape, device = args.backend)
T_norm = np.linalg.norm(value)

d = len(T.shape)
for i in range(d):
    T.prune(axis = i)


# start the simulation
solver = SparseSolver(backend = args.backend)
err_sid = np.zeros((args.nrun, len(args.ranks)))

if args.seed is not None:
    np.random.seed(args.seed)

seeds = np.random.randint(100000000, size = args.nrun)

for i in range(args.nrun):
    print(f'test {i+1}/{args.nrun}')
    for (j, rank) in enumerate(args.ranks):
        print(f'\trank = {rank}')
        idx_sets, sates = solver.fit(
            T, 
            [rank for _ in range(d)], 
            selection_rule = args.selection_rule,
            seed = seeds[i]
        )
        order = [2, 0, 1]
        contract = []
        for k in range(len(order)):
            contract.append(sates[order[k]])
        err = solver.solve_core(T, contract, axis = order, compute_error = True, compute_core = False)
            
        err_sid[i, j] = err / T_norm
        
        print(f'\t\tSatID done, err = {err_sid[i, j]:.3f}')


# visualize

import matplotlib.pyplot as plt
err_sid_mean = np.mean(err_sid, axis = 0)
plt.figure(figsize = (8, 8))
plt.plot(args.ranks, err_sid_mean, 'r--x', label = 'SatID-ave')
plt.legend(loc = 'best')
plt.grid(True)
plt.show()






