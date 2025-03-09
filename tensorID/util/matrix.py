class NormBased:
    
    def __init__(self, backend = 'cpu'):
        self.backend = backend
        if backend == 'cpu':
            import numpy as np
            from scipy.linalg import solve_triangular
            self.lib = np
            self.solve_triangular = solve_triangular
        else:
            try:
                import cupy as cp
                from cupyx.scipy.linalg import solve_triangular
            except:
                raise ValueError(f'got backend {backend} but GPU library cupy is not available.')
            self.lib = cp
            self.solve_triangular = solve_triangular
        
    
    
    def select(self, A, k, rule, seed = None):
        n, d = A.shape
        X = self.lib.zeros((k, d))
        J = []
        if seed is not None:
            self.lib.random.seed(seed)
            
        if rule == 0:
            # uniform
            J = self.lib.random.choice(self.lib.arange(d), replace = False, size = k)
            out = self.lib.linalg.lstsq(A[:, J], A)
            X = out[0]
            score = out[1]  # squared residuals of each col in A
            
        else:
            score = self.lib.linalg.norm(A, axis = 0)**2
            Q = self.lib.zeros((n, k))
            R = self.lib.zeros((k, k))
            for i in range(k):
                if rule == 'inf':
                    # greedy maximal score
                    idx = int(self.lib.argmax(score))
                else:
                    rem = self.lib.argwhere(score > 0).reshape(-1)
                    p = (temp := score[rem]**rule) / self.lib.sum(temp)
                    idx = int(self.lib.random.choice(rem, size = 1, p = p))
                J.append(idx)
                
                # update Q, R and solve for the LS weights w
                u = Q[:, :i].T @ A[:, idx]
                q = A[:, idx] - Q[:, :i] @ u
                v = self.lib.linalg.norm(q)
                q /= v
                Q[:, i] = q
                R[:i, i] = u
                R[i, i] = v
                
                # update X
                X[i] = A.T @ (q / v)
                if i > 0:
                    X[:i] -= self.lib.outer(self.solve_triangular(R[:i, :i], u), X[i])
                
                # update scores for next selection
                score -= (A.T @ Q[:, i])**2 
                score[score < 0] = 0
        return J, X
    

class Nuclear:
    
    def __init__(self, backend = 'cpu'):
        self.backend = backend
        if backend == 'cpu':
            import numpy as np
            from scipy.linalg import solve_triangular
            self.lib = np
            self.solve_triangular = solve_triangular
        else:
            try:
                import cupy as cp
                from cupyx.scipy.linalg import solve_triangular
            except:
                raise ValueError(f'got backend {backend} but GPU library cupy is not available.')
            self.lib = cp
            self.solve_triangular = solve_triangular
    
    
    def select(self, A, k, **kwargs):
        # check if K should be formed explicitly
        m, n = A.shape
        if m > n/2:
            # tall-thin or close to square, compute K
            return self.explicit(A, k, **kwargs)
        else:
            # very short-fat, do not compute K
            return self.implicit(A, k, **kwargs)
    
    
    def explicit(self, A, k, **kwargs):
        n = A.shape[1]
        d = self.lib.linalg.norm(A, axis = 0)**2
        ratio = self.lib.sqrt(min(A.shape) / self.lib.sum(d))
        d *= ratio**2
        A = A * ratio
        
        K = A.T @ A
    
        w = self.lib.linalg.norm(K, axis = 0)**2
        U = self.lib.zeros((k, n))
        S = self.lib.zeros((n, k))
        I = []
        
        for i in range(k):
            d2 = d.copy()
            d2[d2 <= 1E-10] = 1 # protect against 0/0
            scores = w / d2
            scores[I] = 0
            l = int(self.lib.argmax(scores))
            
            I.append(l)
            Ic = [i for i in range(n) if i not in I]

            U[i, l] = 1
            U[:, l] /= self.lib.sqrt(d[l])
            S[:, i] = -K[:, I] @ U[:i+1, l]
            d -= S[:, i]**2
            d[d < 0] = 0
            U[:i+1, Ic] += self.lib.outer(U[:i+1, l], S[Ic, i])
            w += S[:, i] * (2 * S[:, :i+1] @ (S[:, :i+1].T @ S[:, i]) - 2 * K @ S[:, i] - S[:, i] * (S[:, i].T @ S[:, i]))
        
        X = U[:, I] @ S.T
        X *= X[0, I[0]]
        return I, X
    
    
    def implicit(self, A, k, **kwargs):
        # very short-fat, do not compute K
        d = self.lib.linalg.norm(A, axis = 0)**2
        
        # ||A||_F^2 = sum d, normalize this sum to rank
        ratio = self.lib.sqrt(min(A.shape) / self.lib.sum(d))
        A = A * ratio
        d *= ratio**2
        
        # for A = LQ.T, K = Q L.T L Q.T, we diagonalize L.T L
        # K = U D U.T, K2 = U D2 U.T, then compute diag
        #? m2 n instead of m n2
        Q, R = self.lib.linalg.qr(A.T, mode = 'reduced')
        core = R @ R.T
        D, V = self.lib.linalg.eigh(core)
        Q = Q @ V
        w = self.lib.einsum('ij, j, ij -> i', Q, D**2, Q, optimize = 'optimal')
        
        n = A.shape[1]
        #U = self.lib.zeros((k, n))
        UI = self.lib.zeros((k, k))
        LI = self.lib.zeros((k, k))
        S = self.lib.full((n, k), self.lib.nan)
        KI = self.lib.zeros((n, k))
        I = []
        
        #! for debug
        #K = A.T @ A
        
        for i in range(k):
            d2 = d.copy()
            d2[d2 <= 1E-10] = 1 # protect against 0/0
            scores = w / d2
            scores[I] = 0
            
            #top = self.lib.flip(self.lib.argsort(scores))[:5]
            
            l = int(self.lib.argmax(scores))
            
            I.append(l)
            #Ic = [i for i in range(n) if i not in I]
            
            #! for debug
            '''res = K - K[:, I] @ self.lib.linalg.inv(K[I][:, I]) @ K[I, :]
            dd = self.lib.diag(res)
            dd2 = dd.copy()
            dd2[dd2 <= 1E-10] = 1
            ww = self.lib.linalg.norm(res, axis = 0)**2
            ww2 = ww.copy()
            ww2[ww2 <= 1E-10] = 0
            ss = ww2 / dd2
            ttop = self.lib.flip(self.lib.argsort(ss))[:5]'''

            # compute the new column going into KII, store as (u, z), z in R
            newcol = A.T @ A[:, l]
            uz = newcol[I]  #? 2mn extra cost
            KI[:, i] = newcol
            z = uz[-1]
            u = uz[:-1]
            
            # update LI and UI
            if i == 0:
                LI[0, 0] = z**(1/2)
            else:
                LI[i, :i] = self.solve_triangular(LI[:i, :i], u, lower = True)
                LI[i, i] = (z - self.lib.linalg.norm(LI[i, :i])**2)**(1/2) 
            UI[i, i] = 1 / LI[i, i]
            UI[:i, i] = (-1) * UI[i, i] * UI[:i, :i] @ LI[i, :i].reshape(-1)
            #UI[i, i] = 1
            #UI[i, i] /= (Kll - self.lib.linalg.norm(S[l, :i])**2)**(-1/2)
            S[:, i] = KI[:, :i+1] @ UI[:i+1, i]
            d -= S[:, i]**2
            d[d < 0] = 0
            #U[:i+1, Ic] += self.lib.outer(U[:i+1, l], S[Ic, i])
            
            # compute K @ S[:, i] to update w
            KSi = A.T @ (A @ S[:, i])  #? 2mn rather than n2
            w += S[:, i] * (2 * S[:, :i+1] @ (S[:, :i+1].T @ S[:, i]) - 2 * KSi - S[:, i] * (S[:, i].T @ S[:, i]))
        
        X = UI @ S.T
        X *= self.lib.sign(X[0, I[0]])
        return I, X
        
        
        