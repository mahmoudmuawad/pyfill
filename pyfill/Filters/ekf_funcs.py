# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:15:57 2019

@author: mmoawad
"""
import numpy as np
import matplotlib.pyplot as plt
import supp

# Monte carlo check
def monteCarloPrediction(mu,sigma,u_t,input_noise,processModel,number_of_samples):
    n= len(mu)
    mu=np.matrix(mu)            
    sigma=np.matrix(sigma)
    u_t= np.matrix(u_t)
    input_noise= np.matrix(input_noise)
    
    mu_sampled= np.matrix(np.zeros((n,number_of_samples)))
    u_t_sampled= np.matrix(np.zeros((n,number_of_samples)))
    
    for i in range (0,number_of_samples):
        mu_sampled[:,i] = np.matrix(np.random.multivariate_normal(np.array(mu).flatten(), sigma, 1)).T       
        u_t_sampled[:,i] = np.matrix(np.random.multivariate_normal(np.array(u_t).flatten(), input_noise, 1)).T
    #print(mu_sampled)
    
    propagated_sigma_points = np.matrix(np.zeros((n, number_of_samples)))
    
    for i in range(number_of_samples) :
        u = u_t_sampled[:, i]
        #B = np.matrix([[cos(u[0]), 0],
        #      [sin(u[0]), 0]])
        propagated_sigma_points[: ,i] = processModel(mu_sampled[:, i], u)
        
    #propagated_sigma_points=processModel(mu_sampled,u_t_sampled)
    monte_sigma = np.cov(propagated_sigma_points)
    monte_mean = np.matrix(np.zeros((n,1)))

    for j in range (0,n):
        monte_mean[j,0]= np.mean(propagated_sigma_points[j,:])
    
    return monte_mean,monte_sigma
    # linear Merge check
def linearMerge(state_mean,state_cov,measurement,measurement_noise):
    state_mean=np.matrix(state_mean)
    state_cov=np.matrix(state_cov)
    measurement= np.matrix(measurement)
    measurement_noise= np.matrix(measurement_noise)

    kalman_gain = state_cov * (state_cov + measurement_noise).I

    v= measurement-state_mean

    state_updated= state_mean+kalman_gain* v
    cov_updated= (np.matrix(np.eye(2))-kalman_gain)*state_cov

    return state_updated,cov_updated
    
def plot_state(x,cov, x2, cov2, x3, cov3, state_color, state_color_2, state_color_3) :
    fig = plt.figure('x-y plane')
    #plt.axis([-15, 15, -15, 15])
    plt.grid(True)
    ax = fig.add_subplot(1,1,1)
    fig.set_size_inches(8.5, 8.5)
    ax.set_aspect('equal')
    
    supp.DrawPoseWithEllipse(ax,x,cov,state_color,state_color)
    supp.DrawPoseWithEllipse(ax,x2,cov2,state_color_2, state_color_2)
    supp.DrawPoseWithEllipse(ax,x3,cov3,state_color_3, state_color_3)
    plt.show()
    plt.close()
