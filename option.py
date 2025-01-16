#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 23:58:07 2025

@author: hamza
"""

from abc import ABC
import numpy as np
from scipy.stats import norm

class option(ABC):
    def __init__(self):
        pass
    
      
class call(option):
    def __init__(self,K,T):
        self.K = K
        self.T = T
        
        
    def payoff(self,St):
        return np.maximum(St-self.K,0)
        
    def McPrice(self,model):
        st = model.St(self.T)
        return  np.exp(-self.T*model.r)*self.payoff(st).mean()
    
    def BsPrice(self,t,Bsmodel):
        """
        Price a European option using the Black-Scholes formula.
        
        Parameters:
        S : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to maturity (in years)
        r : float
            Risk-free interest rate (annualized)
        sigma : float
            Volatility of the underlying asset (annualized)
            
        Returns:
        float
            The Black-Scholes price of the option
        """
        if self.T <= 0:
            raise ValueError("Time to maturity (T) must be positive.")
        
        # Calculate d1 and d2
        d1 = (np.log(Bsmodel.S0 / self.K) + (Bsmodel.r + 0.5 * Bsmodel.sigma**2) * self.T) / (Bsmodel.sigma * np.sqrt(self.T))
        d2 = d1 - Bsmodel.sigma * np.sqrt(self.T)
        
        # Call option price
        price = Bsmodel.S0 * norm.cdf(d1) - self.K * np.exp(-Bsmodel.r * self.T) * norm.cdf(d2)
        
        return price
    
    
    
class put(option):
    def __init__(self,K,T):
        self.K = K
        self.T = T
        
        
    def payoff(self,St):
        return np.maximum(self.K-St,0)    
    

    def McPrice(self,model):
        st = model.St(self.T)
        return  np.exp(-self.T*model.r)*self.payoff(st).mean()
    
    def BsPrice(self,t,Bsmodel):
        """
        Price a European option using the Black-Scholes formula.
        
        Parameters:
        S : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to maturity (in years)
        r : float
            Risk-free interest rate (annualized)
        sigma : float
            Volatility of the underlying asset (annualized)
            
        Returns:
        float
            The Black-Scholes price of the option
        """
        if self.T <= 0:
            raise ValueError("Time to maturity (T) must be positive.")
        
        # Calculate d1 and d2
        d1 = (np.log(Bsmodel.S0 / self.K) + (Bsmodel.r + 0.5 * Bsmodel.sigma**2) * self.T) / (Bsmodel.sigma * np.sqrt(self.T))
        d2 = d1 - Bsmodel.sigma * np.sqrt(self.T)
        
        # Put option price
        price = self.K * np.exp(-Bsmodel.r * self.T) * norm.cdf(-d2) - Bsmodel.S0 * norm.cdf(-d1)

        return price
    
    
    
    
if __name__ == "__main__":
    pass