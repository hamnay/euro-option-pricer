#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:52:48 2025

@author: hamza
"""

from option import *
from marketModel import *


if __name__ == "__main__" :
    r = 0.02
    sigma = 0.2
    S0 = 1
    T = 5
    K = 1
    n = 9_000_000
    
    BS = blackScholes( r, sigma, S0, T, n)
    
    Call = call( K, T)
    
    st = BS.St(T)

    print(f"Call Price using MC method: {Call.McPrice(BS)}")
    print(f"Call Price using closed form formula: {Call.BsPrice(0,BS)}")
    
    
    
    
    
    
    
    