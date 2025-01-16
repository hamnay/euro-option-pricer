#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:16:55 2025

@author: hamza
"""

import numpy as np 
from typing import Union, List  
from abc import ABC, abstractmethod  




class marketModel(ABC):
    """
    Abstract base class for market models.
    Provides a structure for defining market models with essential methods.
    """

    def __init__(self):
        pass  # No specific initialization for the abstract class

    @abstractmethod
    def St(self, *args, **kwargs):
        """
        Abstract method to compute stock prices.
        Must be implemented by subclasses.
        """
        pass

    def Wt(self, n_paths: int, t: Union[float, List[float]], seed: int = None) -> np.ndarray:
        """
        Generate Brownian motion paths for the given time discretization.

        Parameters:
            n_paths (int): Number of Brownian motion paths.
            t (Union[float, List[float]]): Single time point or list of time points.
            seed (int, optional): Random seed for reproducibility.

        Returns:
            np.ndarray: Array of shape (n_paths, len(t)) representing Brownian motion paths.
        """
        if seed is not None:
            np.random.seed(seed)  # Set seed for reproducibility

        if isinstance(t, (float, int)):  # Single time point
            dt = float(t)  # Ensure time is treated as a float
            sqrt_dt = np.sqrt(dt)  # Compute square root of the time step
            return np.random.normal(loc=0, scale=sqrt_dt, size=n_paths)  # Generate paths
        else:  # Multiple time points
            t = np.array([0] + t, dtype=float)  # Ensure initial time 0 is included
            dt = np.diff(t)  # Compute time intervals
            sqrt_dt = np.sqrt(dt)  # Precompute square root of intervals
            increments = np.random.normal(loc=0, scale=sqrt_dt, size=(n_paths, len(dt)))  # Generate increments
            paths = np.zeros((n_paths, len(t)))  # Initialize paths with zeros
            paths[:, 1:] = np.cumsum(increments, axis=1)  # Cumulative sum for paths
            return paths

class blackScholes(marketModel):
    """
    Implements the Black-Scholes market model for stock price simulation.
    """

    def __init__(self, r: float, sigma: float, S0: float, T: float, n: int = 1000):
        """
        Initialize the Black-Scholes model parameters.

        Parameters:
            r (float): Risk-free interest rate.
            sigma (float): Volatility of the underlying asset.
            S0 (float): Initial stock price.
            T (float): Total time to maturity.
            n (int): Number of simulation paths.
        """
        self.r = r
        self.sigma = sigma
        self.S0 = S0
        self.T = T
        self.n = n

    def St(self, t: Union[float, List[float]], seed: int = None) -> np.ndarray:
        """
        Simulate stock prices using the Black-Scholes formula.

        Parameters:
            t (Union[float, List[float]]): Single time point or list of time points.
            seed (int, optional): Random seed for reproducibility.

        Returns:
            np.ndarray: Simulated stock prices for each path and time point.
        """
        if isinstance(t, (float, int)):  # Single time point
            Wt = self.Wt(self.n, t, seed)  # Generate Brownian motion
            return self.S0 * np.exp((self.r - 0.5 * self.sigma**2) * t + self.sigma * Wt)
        else:  # Multiple time points
            t_array = np.array(t, dtype=float)  # Convert to numpy array
            brownian_paths = self.Wt(self.n, t_array, seed)  # Generate Brownian paths
            exponent = (self.r - 0.5 * self.sigma**2) * t_array + self.sigma * brownian_paths
            return self.S0 * np.exp(exponent)  # Compute stock prices

if __name__ == "__main__":
    # Example usage: Simulate stock prices using Black-Scholes model
    BS = blackScholes(r=0.02, sigma=0.2, S0=1, T=5, n=5)  # Initialize model with parameters
    st = BS.St([2, 5, 8])  # Simulate stock prices at times 2, 5, and 8
    print("Simulated stock prices at t=[2, 5, 8]:\n", st)  # Display results
