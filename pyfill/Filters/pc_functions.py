import numpy as np
from math import *
import random

## can be changed , for my filter i maked it based on normal distribution 
def particles_sample(x, std, N) :   # for x : states , p : standard deviation , senR
    nx=x.shape[0]
    pcs = np.matrix(np.zeros((nx, N)))
    for i in range(nx) :
        pcs[i, :] = np.random.normal(float(x[i]), float(std[i]),(1,N))
    return pcs


def weights_sample(N) :
    weights = np.matrix(np.zeros((1, N)))
    weights[0, :] = 1
    return weights

def pre_sigmas(sig_mat, n_x, pre_model) :
    yi = np.matrix(np.zeros((n_x, sig_mat.shape[1])))
    for i in range(sig_mat.shape[1]) :  
        yi[:, i] = pre_model(sig_mat[0:n_x, i], sig_mat[n_x:, i])
    return yi

def pre_particles(particles, pre_model, u) :
    nx=particles.shape[0]
    N=particles.shape[1]
    out_mat=np.matrix(np.zeros((nx, N)))
    for i in range(N) :
        out_mat[:, i]=pre_model(particles[:,i], u)
    return out_mat

def add_noise(particles, std) :
    nx=particles.shape[0]
    N=particles.shape[1]
    out_mat=np.matrix(np.zeros((nx, N)))
    for i in range(nx): 
        out_mat[i, :]=particles[i, :]+np.random.normal(0, float(std[i]), (1, N))
    return out_mat

def Gaussian(mu, sigma, x):
    # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
    return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))

def measure_prob(weights ,pcs_pred_noise ,measurement, sense_std):
    # calculates how likely a measurement should be
    nx=pcs_pred_noise.shape[0]
    N=pcs_pred_noise.shape[1]
    for i in range(N):
        #print(weights[0, i])
        for j in range(nx):
            weights[0,i] *= Gaussian(pcs_pred_noise[j, i], sense_std[j], measurement[j])
            #print(Gaussian(pcs_pred_noise[j, i], sense_std[j], measurement[j]))
        #print(weights[0,i])
            
    return weights

def resample(weights, particles): 
    nx=particles.shape[0]
    N=particles.shape[1]
    out_mat=np.matrix(np.zeros((nx,N)))
    index = int(random.random() * N)
    beta = 0.0
    mw = np.max(weights)
    for i in range(N):
        beta += random.random() * 2.0 * mw
        #print(weights[0,index])
        while beta > weights[0,index] :
            beta -= weights[0,index]
            index = (index + 1) % N
        out_mat[:,i]=particles[:,index]
    return out_mat