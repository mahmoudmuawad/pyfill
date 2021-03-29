# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:15:00 2019

@author: mmoawad
"""

from math import *
import numpy as np
import supp
from ekf_funcs import *
import matplotlib.pyplot as plt

    
def Extended_KF_predict(x, P, u, Q, compute_f, compute_fj) : 
    F = compute_fj(x, u)        
    x_pred = compute_f(x, u)
    P_pred = F * P * F.T + Q    
    return x_pred, P_pred
    
def Extended_KF_update(x_pred, P_pred, z, R, compute_h, compute_H1, compute_H2) : 
    h = compute_h(x_pred, z)
    H1 = compute_H1(x_pred, z)
    H2 = compute_H2(x_pred, z)        
    Pvv = H1 * P_pred * H1.T + H2 * R * H2.T 
    Pxv = P_pred * H1.T        
    K = Pxv * np.linalg.inv(Pvv)        
    x_update = x_pred - K * h
    P_update = P_pred - K * H1 * P_pred        
    return x_update, P_update

