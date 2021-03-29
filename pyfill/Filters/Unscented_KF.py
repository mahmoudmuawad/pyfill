from math import *
import numpy as np
import supp
#from scipy.linalg import cholesky
import matplotlib.pyplot as plt
from numpy import matrix
from numpy import linalg
from ukf_funcs import *
import ukf
    
def Unscented_KF_predict(x, P, u, Q, pre_model) :
    n_x = x.shape[0]
    n_u = u.shape[0]
    
    x_joint = join_2_vec(x, u)
    P_joint = join_2_mat(P, Q)
    # compute sigma points and weights 
    x_sig = find_sigmas(x_joint, P_joint)
    wi = find_weights(x_joint)
    
    # transform 
    yi = pre_sigmas(x_sig,n_x , pre_model)
    
    # recover gaussian
    x_pred = resample_mean(yi, wi)
    P_pred = resample_cov(yi, wi, x_pred)

    return x_pred, P_pred
    
def Unscented_KF_update(x_pred, P_pred, z, R, compute_h) :
    n_x = x_pred.shape[0]
    n_z = z.shape[0]
    
    z_joint = join_2_vec(x, z)
    P_joint = join_2_mat(P_pred, R)
    
    z_sig = find_sigmas(z_joint, P_joint)
    wj = find_weights(z_joint)
    
    vj = update_sigmas(z_sig, n_x, n_z, compute_h)
    
    v_m = resample_mean(vj, wj)
    Pvv = resample_cov(vj, wj, v_m)
    Pxv = resample_cov_2(z_sig, vj, wj, x_pred, v_m)
    
    K =  Pxv * np.linalg.inv(Pvv)
    
    x_update = x_pred - K*v_m
    P_update = P_pred - K * np.transpose(Pxv)
    
    return x_update, P_update
    



if __name__ == "__main__":
    
    ##################### testing functions ########################
    def ukf_predict(joined_cov, joined_mean):
        xi,w = CalcWeightsAndSigmaPts(joined_cov, joined_mean, 3-4)
        
        m,n = xi.shape
        xi_k1 = np.asmatrix(np.zeros((2,n)))
        
        for i in range(n):
            xi_k1[:,i]= trans_rot(xi[0:2,i], xi[2:,i])
        
        mean1, cov1 = CalcMeanAndCovFromSigmaPts(xi_k1, w)
        
        return (mean1,cov1)
    
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
    
    
    ##################### Linear case ######################3
    def ROT(u_theta) : 
        rt = np.matrix([[cos(u_theta), sin(u_theta)],
                       [-sin(u_theta), cos(u_theta)]])
        return rt
    
    def pre_model(states, u) : 
        B = np.matrix([[cos(u[1]), 0.],
                        [sin(u[1]), 0.]])
        pre_states = ROT(u[1]) * ((states) - B * u)
        return pre_states
    
    def compute_h(x1, x2) :
        return x1 - x2
    
    ############# testing prediction ####################
    # my test 
    
    print("\n######################## Linear Point to Point ##########################\n")    
    
    
    # test prediction to go to object position 
    x = np.matrix(np.zeros((2 ,1)) + 0.5)
    p = np.array([[2., 0.],
                  [0., 2.]])
    u = np.matrix([[0.5*sqrt(2)],
                  [0.25*pi]])
    
    Q = np.array([[0.1, 0.0],
                  [0.0, 0.01*pi]])
    x_monte, p_monte = monteCarloPrediction(x, p, u, Q, pre_model, 2000)
    
    # Note : an important way to test predict must have u values
    
    x_pred,p_pred = Unscented_KF_predict(x, p, u, Q, pre_model)
    
    print("\nfilter prediction : \n")
    print(x_pred)
    print(p_pred)
    print("\nmonte carlo prediction : \n")
    print(x_monte)
    print(p_monte)
    
    x = np.matrix(np.zeros((2 ,1)) + 0.5)
    p = np.array([[2., 0.],
                  [0., 2.]])
    plot_state(x, p ,x_pred, p_pred, x_monte, p_monte, 'k','b', 'r')    
    
    ############# testing update ####################
    
    z = np.matrix([[1.012],
                   [1.03]])
    R = np.matrix([[0.5, 0],
                   [0, 0.2]])    
    x = np.matrix(np.zeros((2 ,1)) + 0.5)
    p = np.array([[0.5, 0],
                  [0, 0.7]])
        
    x_up, p_up = Unscented_KF_update(x, p, z, R, compute_h)
    print("m x update : \n", x_up)
    print("m p update : \n", p_up)
    
    z = np.matrix([[1.012],
                   [1.03]])
    R = np.matrix([[0.5, 0],
                   [0, 0.2]])    
    x = np.matrix(np.zeros((2 ,1)) + 0.5)
    p = np.array([[0.5, 0],
                  [0, 0.7]])
    x_lin, p_lin = linearMerge(x, p, z, R)
    print("l x update : \n", x_lin)
    print("l p update : \n", p_lin)
    
    plot_state(x, p ,x_up, p_up, x_lin, p_lin, 'k','b', 'r')
    