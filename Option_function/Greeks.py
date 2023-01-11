import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import mpl_toolkits.mplot3d.axes3d as plt_3d
import warnings
warnings.filterwarnings('ignore')


class Option_Greeks():

    def __init__(self,S0,K,r,t,T,sigma):
        self.S0=S0
        self.K=K
        self.r=r
        self.t=t
        self.T=T
        self.sigma=sigma

    def Delta(self,Type):
        d1 = (np.log(self.S0 / self.K) + (self.r + self.sigma ** 2 / 2) * (self.T - self.t)) / (self.sigma * np.sqrt(self.T - self.t))
        if Type == 'call':
            delta = ss.norm.cdf(d1)
        elif Type == 'put':
            delta = -(1 - ss.norm.cdf(d1))
        else:
            print('Type Error')
        return delta

    def Gamma(self,Type):
        d1 = (np.log(self.S0 / self.K) + (self.r + self.sigma ** 2 / 2) * (self.T)) / (self.sigma * np.sqrt(self.T))
        pdf = np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi)
        gamma = pdf / (self.S0 * self.sigma * np.sqrt(self.T - self.t))
        return gamma

    def Theta(self, Type):
        d1 = (np.log(self.S0 / self.K) + (self.r + self.sigma ** 2 / 2) * (self.T)) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        pdf = np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi)
        if Type == 'call':
            theta = -(self.S0 * pdf * self.sigma) / (2 * np.sqrt(self.T - self.t)) - self.r * np.exp(-self.r * (self.T - self.t)) * ss.norm.cdf(d2)
        elif Type == 'put':
            theta = -(self.S0 * pdf * self.sigma) / (2 * np.sqrt(self.T - self.t)) + self.r * np.exp(-self.r * (self.T - self.t)) * ss.norm.cdf(-d2)
        return theta

    def Vega(self,Type):
        d1 = (np.log(self.S0 / self.K) + (self.r + self.sigma ** 2 / 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        pdf = np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi)
        vega = self.S0 * pdf * np.sqrt(self.T - self.t)
        return vega

    def Rho(self,Type):
        d1 = (np.log(self.S0 / self.K) + (self.r + self.sigma ** 2 / 2) * (self.T)) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if Type == 'call':
            rho = self.K * (self.T - self.t) * np.exp(-self.r * (self.T - self.t)) * ss.norm.cdf(d2)
        elif Type == 'put':
            rho = -self.K * (self.T - self.t) * np.exp(-self.r * (self.T - self.t)) * ss.norm.cdf(-d2)
        return rho