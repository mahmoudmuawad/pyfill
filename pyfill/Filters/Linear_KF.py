"""
this implementation taken from Probalistic robotics book 
algorithm in page 36 Table 3.1
"""

import numpy as np

def Linear_KF_predict(x, P, u, R, A, B) : 
    x_pred = A * x + B * u
    P_pred = A * P * A.T + R
    
    return x_pred, P_pred

def Linear_KF_update(x_pred, P_pred, z, Q, C) : 
    K = P_pred * np.transpose(C) * np.linalg.inv(C * P_pred * np.transpose(C) + Q)
    x_update = x_pred + K * (z - C * x_pred)
    P_update = (np.matrix(np.eye(x_pred.shape[0], x_pred.shape[0])) - K * C) * P_pred
    
    return x_update, P_update