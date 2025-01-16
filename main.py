#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:52:48 2025

@author: hamza

Script for pricing European Call Options using Monte Carlo and Black-Scholes formulas.
"""

from option import *
from marketModel import *


if __name__ == "__main__":
    # Parameters
    r = 0.02  # Risk-free rate
    sigma = 0.2  # Volatility of the underlying asset
    S0 = 1  # Initial stock price
    T = 5  # Time to maturity in years
    K = 1  # Strike price
    n = 9_000_000  # Number of Monte Carlo simulations

    # Initialize models
    BS = blackScholes(r, sigma, S0, T, n)
    Call = call(K, T)
    Put = put(K, T)
    
    # Simulate and calculate prices
    st = BS.St(T)  # Simulate stock prices at maturity
    mc_CallPrice = Call.mcPrice(2, BS)
    bs_CallPrice = Call.bsPrice(2, BS)
    
    mc_PutPrice = Put.mcPrice(0, BS)
    bs_PutPrice = Put.bsPrice(0, BS)

    # Display results
    print(f"Call Price using MC method: {mc_CallPrice:.6f}")
    print(f"Call Price using closed-form formula: {bs_CallPrice:.6f}")
    print(f"Put Price using MC method: {mc_PutPrice:.6f}")
    print(f"Put Price using closed-form formula: {bs_PutPrice:.6f}")

    
    
    
    
    
    