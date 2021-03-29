# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 17:23:56 2019

@author: mmoawad
"""
from math import *
import numpy as np
import supp
from scipy.linalg import cholesky
import matplotlib.pyplot as plt

from numpy import matrix
from numpy import linalg
#%matplotlib inline

# filter functions functions 
def join_2_mat(mat1, mat2) :
    n1 = mat1.shape[0]
    n2 = mat2.shape[0]
    ret_mat = np.matrix(np.zeros((n1+n2, n1+n2)))
    ret_mat[0:n1, 0:n1] = mat1
    ret_mat[n1:, n1:] = mat2
    return ret_mat

def join_2_vec(vec1, vec2):
    n1 = vec1.shape[0]
    n2 = vec2.shape[0]
    ret_vec = np.matrix(np.zeros((n1+n2, 1)))
    ret_vec[0:n1, 0] = vec1
    ret_vec[n1:, 0] = vec2
    return ret_vec

def find_sigmas(x, p, k = None): 
    n = x.shape[0]
    sig_mat = np.matrix(np.zeros((n, 2*n + 1)))
    if k is None : 
        k = 3 - n    # for guassians
    p_sqrt = linalg.cholesky(p)
    sig_mat[:,0] = x
    for i in range(1,n+1) : 
        
        # both are right 
        
        #sig_mat[:, 2 * i - 1] =  x  +  sqrt(k + n) *p_sqrt[:, i - 1]
        #sig_mat[:, 2 * i] = x - sqrt(k + n) *p_sqrt[:, i - 1]
        
        # as mentioned in ukf paper 
        sig_mat[:, i] =  x  +  sqrt(k + n) *p_sqrt[:, i - 1]
        sig_mat[:, i+n] = x - sqrt(k + n) *p_sqrt[:, i - 1]
    
    return sig_mat

def find_weights(x, k = None) :
    n = x.shape[0]
    if k is None : 
        k = 3 - n    # k or lamda 
    w = np.matrix(np.zeros((2*n+1,1)))
    w[0,0] = k / (n + k)
    for i in range(1, 2*n+1) : 
        w[i,0] = 1/(2*(n + k))
    return w

'''
def ROT(u_theta) : 
    rt = np.matrix([[cos(u_theta), sin(u_theta)],
                   [-sin(u_theta), cos(u_theta)]])
    return rt

def pre_model(states, u) : 
    B = np.matrix([[cos(u[1]), 0.],
                    [sin(u[1]), 0.]])
    pre_states = ROT(u[1]) * ((states) - B * u)
    return pre_states

'''
def pre_sigmas(sig_mat, n_x ,pre_model) :
    yi = np.matrix(np.zeros((n_x, sig_mat.shape[1])))
    for i in range(sig_mat.shape[1]) :  
        yi[:, i] = pre_model(sig_mat[0:n_x, i], sig_mat[n_x:, i])
    return yi
'''
def compute_h(x1, x2) :
    return x1 - x2
'''

def update_sigmas(sig_mat, n_x, n_z, compute_h) :
    vi = np.matrix(np.zeros((n_z, sig_mat.shape[1])))
    for i in range(sig_mat.shape[1]) :
        vi[:, i] = compute_h(sig_mat[0:n_x, i], sig_mat[n_x:, i])
    return vi

# testing finding sigmas and finding weights 
def resample_mean(xi, wi) : 
    return xi * wi

def resample_cov(xi, wi, x) : 
    P = np.asmatrix(np.zeros((xi.shape[0], xi.shape[0])))
    for i in range(xi.shape[1]) : 
        y_diff = xi[:, i] - x
        P += np.asscalar(wi[i]) * y_diff * y_diff.T
    return P



def resample_cov_2(zi, vj, wj, x, v_m) :
    n_x = x.shape[0]
    Pxv = np.asmatrix(np.zeros((n_x, vj.shape[0])))
    #Pxv = np.asmatrix(np.zeros((1, 1)))
    for i in range(zi.shape[1]) : 
        y_diff = zi[0:n_x, i] - x
        v_diff = vj[:, i] - v_m
        Pxv += np.asscalar(wj[i]) * y_diff * v_diff.T
    return Pxv





# Tarrad functions
def CalcWeightsAndSigmaPts(cov, mean, kappa=0.0):
    if np.isscalar(cov):
        cov = matrix([[cov]])
    if np.isscalar(mean):
        mean = matrix([[mean]])
    
    """ 
    Xi - sigma points
    W - weights
    """
    
    #Dimension of the state
    n = np.size(mean)
    
    #weight for most cases
    W = np.asmatrix(np.full((2*n+1, 1), 0.5/(n+kappa)))
    Xi = np.asmatrix(np.zeros((n, 2*n+1)))
    
    #weight at zero 
    #first row in W and Xi
    W[0] = kappa/(n+kappa)
    Xi[:,0] = mean

    # Here the transpose of the upper triangle is used but it doesn't matter
    U = linalg.cholesky((n+kappa)*cov)
    
    for k in range(1,n+1):
        Xi[:, 2 * k - 1] = mean + U[:, k - 1 ]
        Xi[:, 2 * k] = mean - U[:, k - 1]
    
    return (Xi, W)

def CalcMeanAndCovFromSigmaPts(Xi, W, m= None , n=None):
    
    #returns the matrix dimensions(# rows, # cols)
    if ((m == None) and (n == None)):
        m,n = Xi.shape
    
    X = Xi * W

    P = np.asmatrix(np.zeros((m,m)))
    
    for k in range (n):
        s = (Xi[:,k] - X)
        
        # s transpose is done here as x was transposed before
        P+= np.asscalar(W[k])*s*s.T
        
    return (X, P)





if __name__ == '__main__':
    
    ##########################3 testing join functions ###################################
    print("testing join matrices")
    x = np.matrix([[1],
                   [2]])
    p = np.matrix([[1, 0],
                   [0, 2]])
    u = np.matrix([[3],
                   [4]])
    q = np.matrix([[5, 0],
                   [0, 7]])
    print(join_2_mat(p, q))
    print(join_2_vec(x, u))
    
    
    ##################### testing sigma points and weights #############################
    print("sigma and weights tests \n")
    
    x = np.matrix([[5.7441],
                   [1.3800],
                   [2.2049],
                   [0.5015],
                   [0.3528]])
        
    P = np.matrix([[0.0043, -0.0013, 0.0030, -0.0022, -0.0020],
                  [-0.0013,    0.0077,    0.0011,    0.0071,    0.0060],
                  [0.0030,    0.0011,    0.0054,    0.0007,    0.0008],
                  [-0.0022,    0.0071,    0.0007,    0.0098,    0.0100],
                  [-0.0020,    0.0060,    0.0008,    0.0100,    0.0123]])
        
    
    
    xi_t, wi_t = CalcWeightsAndSigmaPts(P, x, 3-5)
    xi_m = find_sigmas(x, P)
    wi_m = find_weights(x)
    
    print ("tarrad xi\n", xi_t.T)
    print ("moawad xi\n", xi_m.T)
    print ("tarrad wi\n", wi_t)
    print("moawad wi\n", wi_m)
    
    print("m recover mean : \n", resample_mean(xi_m, wi_m))
    print("m recover cov : \n", resample_cov(xi_m, wi_m, resample_mean(xi_m, wi_m)))
    me, co = CalcMeanAndCovFromSigmaPts(xi_m, wi_m)
    
    print("t recover mean : \n", me)
    print("t recover cov : \n", co)
    
    ###################### test pre_model #############################
    print("testing pre_model\n")
    x = np.matrix([[1.],
                   [1.]])
    p = np.matrix([[0.5, 0.],
                   [0., 0.5]])
    u = np.matrix([[1*sqrt(2)],
                   [0.25*pi]])
    q = np.matrix([[0.2, 0.],
                   [0, 0.13]])
    print(pre_model(x, u))  #should be 0,0
    
    ###################### test pre_sigmas ############################
    print("testing pre_sigmas")
    y = pre_sigmas(find_sigmas(join_2_vec(x, u),2 ,join_2_mat(p, q)), pre_model)
    print(y)
    print(y.shape)    
    
    
    
    




