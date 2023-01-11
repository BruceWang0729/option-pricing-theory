import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

class Pricing_Model():

    def __init__(self,S0,K,r,t,T,sigma):
        self.S0=S0
        self.K=K
        self.r=r
        self.t=t
        self.T=T
        self.sigma=sigma

    def Binomial_Tree(self,Type):
        N = 10000
        dT = self.T / N
        u = np.exp(self.sigma * np.sqrt(dT))
        d = 1 / u
        V = np.zeros(N + 1)
        S_t = np.array([(self.S0 * u ** i * d ** (N - i)) for i in range(N + 1)])
        ret = np.exp(self.r * dT)
        p = (ret - d) / (u - d)
        q = 1 - p
        if Type == "call":
            V[:] = np.maximum(S_t - self.K, 0)
        elif Type == 'put':
            V[:] = np.maximum(self.K - S_t, 0)
        else:
            print("The type error")

        for i in range(N - 1, -1, -1):
            V[:-1] = np.exp(-self.r * dT) * (p * V[1:] + q * V[:-1])
        return V[0]

    def Monte_Carl(self,Type):
        N = 1000000
        temp = ss.norm.rvs((self.r - 0.5 * self.sigma ** 2) * self.T, np.sqrt(self.T) * self.sigma, N)
        S_t = self.S0 * np.exp(temp)
        if Type == 'call':
            ret = np.sum(np.exp(-self.r * self.T) * np.maximum(S_t - self.K, 0)) / N
        elif Type == 'put':
            ret = np.sum(np.exp(-self.r * self.T) * np.maximum(self.K - S_t, 0)) / N
        else:
            print('Type Error')
        return ret

    def BS_formula(self,Type):
        d1 = (np.log(self.S0 / self.K) + (self.r + self.sigma ** 2 / 2) * (self.T)) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        if Type == 'call':
            ret = self.S0 * ss.norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * ss.norm.cdf(d2)
        elif Type == 'put':
            ret = -self.S0 * ss.norm.cdf(-d1) + self.K * np.exp(-self.r * self.T) * ss.norm.cdf(-d2)
        return ret
# #
# if __name__=='__main__':
#     model=Pricing_Model(100,100,0.05,0,1,0.2)
#     print(model.Binomial_Tree('call'))
#     print(model.Binomial_Tree('put'))
#     print(model.Monte_Carl('call'))
#     print(model.Monte_Carl('put'))
#     print(model.BS_formula('call'))
#     print(model.BS_formula('put'))