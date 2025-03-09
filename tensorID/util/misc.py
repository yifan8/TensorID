import numpy as np
  
def next_pow2(n):
    if n < 2:
        return 2  # at least 2
    return int(2**int(np.log2(n-0.01) + 1) + 0.01)


def numel(shape):
    return np.prod(np.array(shape, dtype = np.int64))
