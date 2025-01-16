#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 23:58:07 2025

@author: hamza
"""

from abc import ABC
from typing import Any, Union
import numpy as np
from scipy.stats import norm

class option(ABC):
    """
    Abstract base class for financial options.
    """

    def __init__(self):
        pass


class call(option):
    """
    Represents a European call option.
    """

    def __init__(self, K: float, T: float):
        """
        Initialize a call option.

        Parameters:
            K (float): Strike price of the option.
            T (float): Time to maturity in years.
        """
        self.K = K  # Strike price
        self.T = T  # Time to maturity

    def payoff(self, St: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculate the payoff of the call option.

        Parameters:
            St (Union[float, np.ndarray]): Stock price(s) at maturity.

        Returns:
            np.ndarray: Payoff(s) of the call option.
        """
        return np.maximum(St - self.K, 0)

    def mcPrice(self, t: float, model: Any) -> float:
        """
        Calculate the option price using the Monte Carlo method.

        Parameters:
            model (Any): Market model providing stock price simulations and risk-free rate.

        Returns:
            float: Monte Carlo estimate of the option price.
        """
        st = model.St(self.T)  # Simulate stock prices at maturity
        return np.exp(-self.T * model.r) * self.payoff(st).mean()

    def bsPrice(self, t: float, Bsmodel: Any) -> float:
        """
        Calculate the option price using the Black-Scholes formula.

        Parameters:
            t (float): Current time (in years).
            Bsmodel (Any): Black-Scholes market model.

        Returns:
            float: Black-Scholes price of the call option.
        """
        if self.T <= 0:
            raise ValueError("Time to maturity (T) must be positive.")

        # Calculate d1 and d2
        d1 = (np.log(Bsmodel.S0 / self.K) +
              (Bsmodel.r + 0.5 * Bsmodel.sigma**2) * self.T) / \
             (Bsmodel.sigma * np.sqrt(self.T))
        d2 = d1 - Bsmodel.sigma * np.sqrt(self.T)

        # Call option price
        price = Bsmodel.S0 * norm.cdf(d1) - self.K * np.exp(-Bsmodel.r * self.T) * norm.cdf(d2)
        return price


class put(option):
    """
    Represents a European put option.
    """

    def __init__(self, K: float, T: float):
        """
        Initialize a put option.

        Parameters:
            K (float): Strike price of the option.
            T (float): Time to maturity in years.
        """
        self.K = K  # Strike price
        self.T = T  # Time to maturity

    def payoff(self, St: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculate the payoff of the put option.

        Parameters:
            St (Union[float, np.ndarray]): Stock price(s) at maturity.

        Returns:
            np.ndarray: Payoff(s) of the put option.
        """
        return np.maximum(self.K - St, 0)

    def mcPrice(self, t: float, model: Any) -> float:
        """
        Calculate the option price using the Monte Carlo method.

        Parameters:
            model (Any): Market model providing stock price simulations and risk-free rate.

        Returns:
            float: Monte Carlo estimate of the option price.
        """
        st = model.St(self.T)  # Simulate stock prices at maturity
        return np.exp(-(self.T - t) * model.r) * self.payoff(st).mean()

    def bsPrice(self, t: float, Bsmodel: Any) -> float:
        """
        Calculate the option price using the Black-Scholes formula.

        Parameters:
            t (float): Current time (in years).
            Bsmodel (Any): Black-Scholes market model.

        Returns:
            float: Black-Scholes price of the put option.
        """
        if self.T <= 0:
            raise ValueError("Time to maturity (T) must be positive.")

        # Calculate d1 and d2
        d1 = (np.log(Bsmodel.S0 / self.K) +
              (Bsmodel.r + 0.5 * Bsmodel.sigma**2) * self.T) / \
             (Bsmodel.sigma * np.sqrt(self.T))
        d2 = d1 - Bsmodel.sigma * np.sqrt(self.T)

        # Put option price
        price = self.K * np.exp(-Bsmodel.r * self.T) * norm.cdf(-d2) - Bsmodel.S0 * norm.cdf(-d1)
        return price


if __name__ == "__main__":
    # Placeholder for main execution
    pass
