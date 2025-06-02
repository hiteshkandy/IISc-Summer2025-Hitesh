import numpy as np


class RACIPE:
    def __init__(self, topology, n_init=50, t_max=100.0, dt=0.1):
        self.genes = topology['genes']
        self.reg = topology['regulators']
        self.N = len(self.genes)
        self.n_init = n_init
        self.t_max = t_max
        self.dt = dt

    def _sample_intrinsic(self):
        G = np.random.uniform(1, 100)
        k = np.random.uniform(1, 100)
        return G, 1/k

    def _estimate_X0_median(self, n_samples=2000):
        ratios = np.empty(n_samples)
        for i in range(n_samples):
            G, k = self._sample_intrinsic()
            ratios[i] = G / k
        return np.median(ratios)

    def _sample_model_parameters(self, M):
        p = {'G': {}, 'k': {}, 'n': {}, 'lam': {}, 'X0': {}}
        # intrinsic parameters
        for g in self.genes:
            p['G'][g], p['k'][g] = self._sample_intrinsic()
        # regulatory parameters
        for target, regs in self.reg.items():
            for reg, typ in regs:
                n = np.random.randint(1, 7)
                p['n'][(reg, target)] = n
                if typ == 'act':
                    lam = np.random.uniform(1, 100)
                else:
                    lam = 1.0 / np.random.uniform(1, 100)
                p['lam'][(reg, target)] = lam
                X0 = np.random.uniform(0.02 * M[reg], 1.98 * M[reg])
                p['X0'][(reg, target)] = X0
        return p

    def _shifted_hill(self, x, X0, n, lam):
        return lam + (1 - lam) / (1 + (x / X0)**n)

    def _dynamics(self, y, p):
        dA = np.zeros(self.N)
        for i, g in enumerate(self.genes):
            prod = 1.0
            # combine regulator effects
            for reg, typ in self.reg.get(g, []):
                j = self.genes.index(reg)
                hs = self._shifted_hill(y[j], p['X0'][(reg, g)], p['n'][(reg, g)], p['lam'][(reg, g)])
                prod *= hs
            # compute basal to cap at G
            act_lams = [p['lam'][(reg, g)] for reg, typ in self.reg.get(g, []) if typ == 'act']
            prod_max = np.prod(act_lams) if act_lams else 1.0
            g_basal = p['G'][g] / prod_max
            dA[i] = g_basal * prod - p['k'][g] * y[i]
        return dA

    def _rk4_step(self, y, p):
        dt = self.dt
        k1 = self._dynamics(y, p)
        k2 = self._dynamics(y + dt * k1 / 2, p)
        k3 = self._dynamics(y + dt * k2 / 2, p)
        k4 = self._dynamics(y + dt * k3, p)
        return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

    def simulate_one(self, args):
        topology, M = args
        p = self._sample_model_parameters(M)
        states = []
        for _ in range(self.n_init):
            y = np.array([10.0 ** np.random.uniform(
                 np.log10(p['G'][g]/p['k'][g]) - 2.5,
                 np.log10(p['G'][g]/p['k'][g]) + 2.5
             ) for g in self.genes])
            t = 0.0
            while t < self.t_max:
                y = self._rk4_step(y, p)
                t += self.dt
            states.append(y)
        return np.unique(np.vstack(states), axis=0)

