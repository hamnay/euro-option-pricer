#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:16:55 2025

@author: hamza
"""

import numpy as np
from typing import Union, List
from abc import ABC


class marketmodel(ABC) :
    def __init__(self):
        pass
    
    def Wt(self,n_paths: int, t: Union[float, List[float]], seed: int = None) -> np.ndarray:
        """
        Generate Brownian motions with given discretization points.
    
        Parameters:
            n_paths (int): Number of Brownian motion paths to generate.
            t (Union[float, List[float]]): Total time duration as a float or a list of discretization points.
            seed (int, optional): Random seed for reproducibility.
    
        Returns:
            np.ndarray: Array of shape (n_paths, len(t)) representing Brownian motion paths.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Handle t as a float or list
        if isinstance(t, (float, int)):  # Single time point
            dt = float(t)  # Ensure t is treated as a float
            sqrt_dt = np.sqrt(dt)
            return np.random.normal(loc=0, scale=sqrt_dt, size=n_paths)
        else:  # List of time points
            t = np.array([0]+t, dtype=float)
            n_steps = len(t)-1  # Number of intervals
            dt = np.diff(t)
            sqrt_dt = np.sqrt(dt)
            
            # Generate standard normal increments
            increments = np.random.normal(loc=0, scale=sqrt_dt, size=(n_paths, n_steps))
            
            # Prepend zeros for the initial value
            paths = np.zeros((n_paths, len(t)))
            paths[:, 1:] = np.cumsum(increments, axis=1)
            return paths
   
    def St(self):
        pass



class blackScholes(marketmodel):
    def __init__(self,r,sigma,S0,T,n=1000):
        self.n=n
        self.T=T
        self.r=r
        self.sigma=sigma
        self.S0=S0
     
    def St(self, t: Union[float, List[float]], seed: int = None) -> np.ndarray:
        if isinstance(t, (float, int)):
            Wt = self.Wt(self.n, t, seed)
            return self.S0*np.exp((self.r-0.5*self.sigma**2)*t+self.sigma*Wt)
         
        else:
            t_array = np.array(t, dtype=float)
            brownian_paths = self.Wt(self.n, t_array, seed)
            exponent = (self.r - 0.5 * self.sigma**2) * t_array + self.sigma * brownian_paths
            return self.S0 * np.exp(exponent)   



if __name__ == "__main__":
    BS = blackScholes(r = 0.02,sigma = 0.2,S0 = 1,T = 5, n = 5)
    st = BS.St([2,5,8])
    
    