import math
import numpy
import numpy as np
from numpy import matrix
from numpy import linalg
import supp
import matplotlib.pyplot as plt
from supp import trans_rot
from supp import rot_trans
from supp import GetR

#from linear_merge import linear_merge


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
    
def CalcWeightsAndSigmaPtsSqrtCov(cov_sqrt, mean, kappa=0.0):
    if np.isscalar(cov_sqrt):
        cov_sqrt = matrix([[cov_sqrt]])
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
    
    for k in range(1,n+1):
        Xi[:, 2 * k - 1] = mean + (np.sqrt(n+kappa)) * cov_sqrt[:, k - 1]
        Xi[:, 2 * k] = mean - (np.sqrt(n+kappa)) * cov_sqrt[:, k - 1]
    
    return (Xi, W)

#######################################
def CalcStatesAndMeasSigmaPtsSqrtCov(state,meas,Q1,R10,R11,R12,Q2,R20,kappa=0.0):
    if np.isscalar(state):
        state = matrix([[state]])
    if np.isscalar(meas):
        meas = matrix([[meas]])
    if np.isscalar(Q1):
        Q1 = matrix([[Q1]])
    if np.isscalar(R10):
        R10 = matrix([[R10]])
    if np.isscalar(R11):
        R11 = matrix([[R11]])
    if np.isscalar(R12):
        R12 = matrix([[R12]])
    if np.isscalar(Q2):
        Q1 = matrix([[Q2]])
    if np.isscalar(R20):
        R20 = matrix([[R20]])
    
    n_state = np.size(state)
    n_meas = np.size(meas)
    
    n= n_state * 4 + 2 * n_meas
    n_samples= 2 * n +1 
    
    W = np.asmatrix(np.full((n_samples, 1), 0.5/(n+kappa)))
    W[0] = kappa/(n+kappa)
    
    state_sample= np.matrix(np.zeros((n_state,n_samples)))
    meas_sample= np.matrix(np.zeros((n_meas,n_samples)))
    
    state_sample[:,1:n_state+1]= Q1
    state_sample[:,n_state+1:2*n_state+1]= -1 * Q1
    state_sample[:,2*n_state+1:3*n_state+1]= R10
    state_sample[:,3*n_state+1:4*n_state+1]= -1 * R10
    state_sample[:,4*n_state+1:5*n_state+1]= R11
    state_sample[:,5*n_state+1:6*n_state+1]= -1 * R11
    state_sample[:,6*n_state+1:7*n_state+1]= R12
    state_sample[:,7*n_state+1:8*n_state+1]= -1 * R12
    
    meas_sample[:,8*n_state+1:8*n_state+1+n_meas]= Q2
    meas_sample[:,8*n_state+1+n_meas:8*n_state+1+2*n_meas]= -1 * Q2
    meas_sample[:,8*n_state+1+2*n_meas:8*n_state+1+3*n_meas]= R20
    meas_sample[:,8*n_state+1+3*n_meas:8*n_state+1+4*n_meas]= -1 * R20
    
    gamma= np.sqrt(n+kappa)
    state_sample= gamma * state_sample
    meas_sample= gamma * meas_sample
    
    state_mean_sample= np.asmatrix(np.full((n_state, n_samples), state))
    meas_mean_sample= np.asmatrix(np.full((n_meas, n_samples), meas))
    
    state_sample= state_sample + state_mean_sample
    meas_sample= meas_sample + meas_mean_sample
    
    return (state_sample,meas_sample, W)

######################################
def StatesAndMeasSigmasToZeroMean(Q1,R10,R11,R12,Q2,R20,kappa=0.0):

    if np.isscalar(Q1):
        Q1 = matrix([[Q1]])
    if np.isscalar(R10):
        R10 = matrix([[R10]])
    if np.isscalar(R11):
        R11 = matrix([[R11]])
    if np.isscalar(R12):
        R12 = matrix([[R12]])
    if np.isscalar(Q2):
        Q2 = matrix([[Q2]])
    if np.isscalar(R20):
        R20 = matrix([[R20]])
    
    n_state = Q1.shape[0]
    n_meas = Q2.shape[0]
    
    n= n_state * 4 + 2 * n_meas
    n_state_samples= n_state * 4  
    n_meas_samples= 2 * n_meas
    n_samples= 2 * n +1 
    
    W = np.asmatrix(np.full((n_samples, 1), 0.5/(n+kappa)))
    W[0] = kappa/(n+kappa)
    
    state_sample= np.matrix(np.zeros((n_state,n_state_samples)))
    meas_sample= np.matrix(np.zeros((n_meas,n_meas_samples)))
    
    state_sample[:,0:n_state]= Q1
    state_sample[:,n_state:2*n_state]= R10
    state_sample[:,2*n_state:3*n_state]= R11
    state_sample[:,3*n_state:4*n_state]= R12

    
    meas_sample[:,0:n_meas]= Q2
    meas_sample[:,n_meas:2*n_meas]= R20

    
    gamma= np.sqrt(n+kappa)
    state_sample= gamma * state_sample
    meas_sample= gamma * meas_sample
        
    return (state_sample,meas_sample, W)





#######################################

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
    

def CalcMeanAndCovFromSigmaPts_sci(Xi, W, m , n):
    
    n_state = Xi.shape[0]
    mean =  Xi * W
    P = np.asmatrix(np.zeros((n_state,n_state)))
    
    for k in range (m , n):
        s = (Xi[:,k] - mean)
        
        # s transpose is done here as x was transposed before
        P+= np.asscalar(W[k])*s*s.T
        
    return (mean, P)
    
    
def GetSplitMeanAndSqrtCovFromSamplesSci(Xi, W, m , n, r=0, s=0):
    
    m_complete_samples, n_complete_samples = Xi.shape
    mean =  Xi * W
    
    n_splitted_samples1 = (n-m)
    m_splitted_weights1 = (n-m)
    n_splitted_samples2 = (s-r)
    m_splitted_weights2 = (s-r)
    
    
    splitted_samples = np.asmatrix(np.zeros((m_complete_samples, n_splitted_samples1 + n_splitted_samples2)))
    splitted_weights = np.asmatrix(np.zeros((m_splitted_weights1 + m_splitted_weights2, 1)))
    
    cov_sqrt2 = np.asmatrix(np.zeros((m_complete_samples, n_splitted_samples1+n_splitted_samples2)))
    
    splitted_samples[:, 0:n_splitted_samples1] = Xi[:, m:n]
    splitted_samples[:, n_splitted_samples1 : n_splitted_samples1 + n_splitted_samples2] = Xi[:, r:s]
    
    splitted_weights[0:m_splitted_weights1,:] = W[m:n, :]
    splitted_weights[m_splitted_weights1 : m_splitted_weights1 + m_splitted_weights2,:] = W[r:s, :]
    
    for k in range (n_splitted_samples1 + n_splitted_samples2):
        w_sqrt = np.sqrt(np.asscalar(splitted_weights[k]))
        
        # s transpose is done here as x was transposed before
        cov_sqrt2[:,k]= w_sqrt * (splitted_samples[:,k] - mean)
    
    
    R = GetR(cov_sqrt2.T)
        
    return (mean, R.T)
    

def GetSplitSqrtCovFromSamplesAndMeanSci(mean, Xi, W, m , n, r=0, s=0):
    
    m_complete_samples, n_complete_samples = Xi.shape
    
    n_splitted_samples1 = (n-m)
    m_splitted_weights1 = (n-m)
    n_splitted_samples2 = (s-r)
    m_splitted_weights2 = (s-r)
    
    
    splitted_samples = np.asmatrix(np.zeros((m_complete_samples, n_splitted_samples1 + n_splitted_samples2)))
    splitted_weights = np.asmatrix(np.zeros((m_splitted_weights1 + m_splitted_weights2, 1)))
    
    cov_sqrt2 = np.asmatrix(np.zeros((m_complete_samples, n_splitted_samples1+n_splitted_samples2)))
    
    splitted_samples[:, 0:n_splitted_samples1] = Xi[:, m:n]
    splitted_samples[:, n_splitted_samples1 : n_splitted_samples1 + n_splitted_samples2] = Xi[:, r:s]
    
    splitted_weights[0:m_splitted_weights1,:] = W[m:n, :]
    splitted_weights[m_splitted_weights1 : m_splitted_weights1 + m_splitted_weights2,:] = W[r:s, :]
    
    for k in range (n_splitted_samples1 + n_splitted_samples2):
        w_sqrt = np.sqrt(np.asscalar(splitted_weights[k]))
        
        # s transpose is done here as x was transposed before
        cov_sqrt2[:,k]= w_sqrt * (splitted_samples[:,k] - mean)
    
    
    R = GetR(cov_sqrt2.T)
        
    return ( R.T)
    



def CalcMeanAndSqrtCovFromSigmaPts(Xi, W, m= None , n=None):
    
    if ((m == None) and (n == None)):
        m,n = Xi.shape
    
    mean = Xi * W

    cov_sqrt2 = np.asmatrix(np.zeros((m,n)))   
    
    for k in range (n):
        w_sqrt = np.sqrt(np.asscalar(W[k]))
        
        cov_sqrt2[:,k] = w_sqrt * (Xi[:,k] - mean)
    
    R = GetR(cov_sqrt2.T)
    
    return (mean, R.T)
    
    

    

    

def CalcPtsInnovation(Z):
    
    dx = Z[2,0] - Z[0,0]
    dy = Z[3,0] - Z[1,0]
    
    return matrix([[dx],[dy]])
    
    
def GetMean(samples, weights):
    
    return samples * weights
    
# Scale given sample to maintain covariance matrix
def ScaleSample(sample, mean_samples, weight_to_add):
    
    sample_without_mean = (sample - mean_samples)
    sample_without_mean = sample_without_mean / np.sqrt(1.0 - weight_to_add)
    return mean_samples + sample_without_mean
    
# Builds a joint density out of given samples and a gaussian random variable.
# Given samples must be structured as 2 * m + 1 with m > 0. The number joint density
# samples is 2 * (m + n) + 1 with n being the dimension of the gaussian.
# Mean and covariances of samples and gaussian remain the same; cross covariances
# are zero. Erros are introduced starting from the third central moment of
# the joint density, since given samples are not symmetric in general.
def BuildJointDensity(samples, weights, expected_value, cov_sqrt_mat):
    
    n_state = samples.shape[0]
    n_expected_value = expected_value.shape[0]
    n_incomplete = (samples.shape[1] - 1) / 2
    n_complete = n_incomplete + n_expected_value
    num_complete = 2 * (n_complete) + 1
    mean_samples = GetMean(samples, weights)
    non_zero_weight = 1.0 / (2.0 * n_complete)
    weight_to_add = 2.0 * n_expected_value * non_zero_weight
    
    joint_samples = np.matrix(np.zeros((n_state + n_expected_value, num_complete)))
    joint_weights = np.matrix(np.zeros((1, num_complete)))
    
    scaled_sample = ScaleSample(samples[:, 0], mean_samples, weight_to_add)
    
    joint_samples[0 : n_state, 0] = scaled_sample
    joint_samples[-n_expected_value:, 0] = expected_value
    joint_weights[:, 0] = 0.0
    
    for i in range(1, n_incomplete + 1):
        
        # Add first one of sample pair
        scaled_sample = ScaleSample(samples[:, 2 * i - 1], mean_samples,
                                         weight_to_add)
                                         
        joint_samples[0 : n_state, 2 * i - 1] = scaled_sample
        joint_samples[-n_expected_value:, 2 * i - 1] = expected_value
        joint_weights[:, 2 * i - 1] = non_zero_weight
        
        # Add second one of sample pair
        scaled_sample = ScaleSample(samples[:, 2 * i], mean_samples,
                                         weight_to_add)
                                         
        joint_samples[0 : n_state, 2 * i] = scaled_sample
        joint_samples[-n_expected_value:, 2 * i] = expected_value
        joint_weights[:, 2 * i] = non_zero_weight
    
    for i in range(n_incomplete + 1, n_complete + 1):
        
        # Add first one of sample pair
        joint_samples[0 : n_state, 2 * i - 1] = mean_samples
        joint_samples[-n_expected_value:, 2 * i - 1] = expected_value + np.sqrt(n_complete) * cov_sqrt_mat[:, i - 1 - n_incomplete]
        joint_weights[:, 2 * i - 1] = non_zero_weight
        
        # Add second one of sample pair
        joint_samples[0 : n_state, 2 * i] = mean_samples
        joint_samples[-n_expected_value:, 2 * i] = expected_value - np.sqrt(n_complete) * cov_sqrt_mat[:, i - 1 - n_incomplete]
        joint_weights[:, 2 * i] = non_zero_weight
        joint_weights_t = joint_weights.T
    
    return joint_samples, joint_weights_t
    
#To change the motion model -> replace trans_rot() with the desired motion model    
def ukf_predict(joined_cov, joined_mean):
    xi,w = CalcWeightsAndSigmaPts(joined_cov, joined_mean)
    
    m,n = xi.shape
    xi_k1 = np.asmatrix(np.zeros((2,n)))
    
    for i in range(n):
        xi_k1[:,i]= trans_rot(xi[0:2,i], xi[2:,i])
    
    mean1, cov1 = CalcMeanAndCovFromSigmaPts(xi_k1, w)
    
    return (mean1,cov1)

#To change the motion model -> replace trans_rot() with the desired motion model    
def ukf_predict_optimized(joined_cov, joined_mean):
    xi,w = CalcWeightsAndSigmaPts(joined_cov, joined_mean)
    m,n = xi.shape
    xi_k1 = np.asmatrix(np.zeros((2,n)))
    
    for i in range(n):
        trans_rot (xi[:,i], xi[2,i], xi[3,i], xi_k1[:,i])
    
    mean1, cov1 = CalcMeanAndCovFromSigmaPts(xi_k1, w)
    
    return (mean1,cov1,xi_k1,w)
    
# To change the innovation fn -> replace CalcPtsInnovation() with the desired innovation fn 
def ukf_merge(mean1, cov1, mean2, cov2):
    
    #Build Joint state and covariance matrices of the (prediction and measurement)
    Z = np.asmatrix(np.zeros((mean1.shape[0]+mean2.shape[0],1)))
    Z[0:mean1.shape[0]] = mean1
    Z[mean1.shape[0]:] = mean2
    Pz = supp.JoinMatsOnDiag(cov1, cov2)
    
    #Calculate the sigma points of the Joint
    Zi, wi = CalcWeightsAndSigmaPts(Pz, Z)
    
    state_samples = Zi[0:mean1.shape[0] , :]
    meas_samples = Zi[mean1.shape[0]: , :]

    m,n = Zi.shape
    
    #Calculate the innovation sigma points from the joint sigma points
    Vi =  np.asmatrix(np.zeros((2,n)))
    
    for j in range(n):
        Vi[:,j] = supp.pt_meas_model(state_samples[0:2,j], meas_samples[:,j])
    
    #Calculate the mean and covariance of the innovation
    V, Pvv = CalcMeanAndCovFromSigmaPts(Vi, wi)

    n1 = mean1.size
    nv = V.shape[0]
    Pxv = np.asmatrix(np.zeros((n1,nv)))
    
    for k in range (n):
        s1 = (state_samples[:,k] - mean1)
        s2 = (Vi[:,k] - V)
        
        Pxv+= np.asscalar(wi[k])*s1*s2.T
        
    K = - Pxv * (Pvv.I)
    
    Pn = cov1 + K * Pxv.T
    
    Xn = mean1 + K * V
    
    return (Xn,Pn)
    
# To change the innovation fn -> replace CalcPtsInnovation() with the desired innovation fn 
def scaled_ukf_merge(mean1, cov1, mean2, cov2):
    
    #Build Joint state and covariance matrices of the (prediction and measurement)
    Z = np.asmatrix(np.zeros((mean1.shape[0]+mean2.shape[0],1)))
    Z[0:mean1.shape[0]] = mean1
    Z[mean1.shape[0]:] = mean2
    Pz = supp.JoinMatsOnDiag(cov1, cov2)
    
    #Calculate the sigma points of the Joint
    Zi, wi = CalcWeightsAndSigmaPts(Pz, Z)

    m,n = Zi.shape
    
    alpha = 0.2
    
    state_samples = Zi[0:mean1.shape[0],:]
    meas_samples = Zi[mean1.shape[0]:,:]
    
    scaled_state_samples = np.asmatrix(np.zeros((mean1.shape[0],n)))
    unscaled_state_samples = np.asmatrix(np.zeros((mean1.shape[0],n)))
    scaled_meas_samples = np.asmatrix(np.zeros((mean2.shape[0],n)))
    
    for j in range (Zi.shape[1]):
        
        scaled_state_samples[:,j] = ((state_samples[:,j] - state_samples[:,0] ) * alpha) + state_samples[:,0]
        scaled_meas_samples[:,j] = ((meas_samples[:,j] - meas_samples[:,0] ) * alpha) + meas_samples[:,0]
        
    
    
    #Calculate the innovation sigma points from the joint sigma points
    Vi =  np.asmatrix(np.zeros((2,n)))
    Vi0 =  np.asmatrix(np.zeros((2,n)))
    
    for j in range(n):
        Vi[:,j] = supp.pt_meas_model(scaled_state_samples[:,j], scaled_meas_samples[:,j])
        Vi0[:,j] = supp.pt_meas_model(state_samples[:,j], meas_samples[:,j])
    
    
    # unscale Innovation and state samples
    for j in range(Zi.shape[1]):
        Vi[:,j] = ((Vi[:,j] - Vi[:,0])/ alpha**2) + Vi[:,0]
        unscaled_state_samples[:,j] = ((scaled_state_samples[:,j] - scaled_state_samples[:,0])/ alpha**2) + scaled_state_samples[:,0]
    
    #Calculate the mean and covariance of the innovation
    V, Pvv = CalcMeanAndCovFromSigmaPts(Vi, wi)
    
    V0, Pvv0 = CalcMeanAndCovFromSigmaPts(Vi0, wi)
    
    Pvv = alpha**2 * Pvv
    

    nx = mean1.shape[0]
    nv = V.shape[0]
    Pxv = np.asmatrix(np.zeros((nx,nv)))
    Pxv0 = np.asmatrix(np.zeros((nx,nv)))
    
    for k in range (n):
        s1 = (unscaled_state_samples[:,k] - mean1)
        s2 = (Vi[:,k] - V)
        
        Pxv+= np.asscalar(wi[k])*s1*s2.T
        
        s1 = (state_samples[:,k] - mean1)
        s2 = (Vi0[:,k] - V)
        
        Pxv0+= np.asscalar(wi[k])*s1*s2.T
        
    
    
    Pxv = alpha**2 * Pxv 
    
    K = - Pxv * (Pvv.I)
    
    K0 = - Pxv0 * (Pvv0.I)
    
#    print K - K0
#    print
    
    Pn = cov1 + K * Pxv.T
    
    Xn = mean1 + K * V
    
    return (Xn,Pn)
    
# To change the innovation fn -> replace CalcPtsInnovation() with the desired innovation fn 
def ukf_merge_optimized(mean1, cov1, sigma_pts1, wi1, mean2, cov2):
    
    #Build Joint state and covariance matrices of the (prediction and measurement)
    cov2_sqrt = linalg.cholesky(cov2)
    joint_samples, joint_weights = BuildJointDensity(sigma_pts1, wi1, mean2, cov2_sqrt)

    m,n = joint_samples.shape
    
    #Calculate the innovation sigma points from the joint sigma points
    Vi =  np.asmatrix(np.zeros((2,n)))
    
    for i in range(n):
        Vi[:,i] = CalcPtsInnovation(joint_samples[:,i])
    
    #Calculate the mean and covariance of the innovation
    V, Pvv = CalcMeanAndCovFromSigmaPts(Vi, joint_weights)
    
    Xii = joint_samples[0:2,:]

    n1 = mean1.size
    Pxv = np.asmatrix(np.zeros((n1,n1)))
    
    for k in range (n):
        s1 = (Xii[:,k] - mean1)
        s2 = (Vi[:,k] - V)
        
        Pxv+= np.asscalar(joint_weights[k])*s1*s2.T
        
    K = - Pxv * (Pvv.I)
    
    Pn = cov1 + K * Pxv.T
    
    Xn = mean1 + K * V
    
    return (Xn,Pn)
    


#To change the motion model -> replace trans_rot() with the desired motion model    
def iesukf_predict(joined_cov_sqrt, joined_state):
    
    xi,w = CalcWeightsAndSigmaPtsSqrtCov(joined_cov_sqrt, joined_state)
    
    m,n = xi.shape
    xi_k1 = np.asmatrix(np.zeros((2,n)))
    
    for i in range(n):
        trans_rot (xi[:,i], xi[2,i], xi[3,i], xi_k1[:,i])
    
    mean1, cov_sqrt1 = CalcMeanAndSqrtCovFromSigmaPts(xi_k1, w)
    
    return (mean1,cov_sqrt1,xi_k1,w)
    

# To change the innovation fn -> replace CalcPtsInnovation() with the desired innovation fn 
def iesukf_merge(mean1, cov1, sigma_pts1, wi1, mean2, cov2):
    
    #Build Joint state and covariance matrices of the (prediction and measurement)
    cov2_sqrt = linalg.cholesky(cov2)
    
    joint_samples, joint_weights = BuildJointDensity(sigma_pts1, wi1, mean2, cov2_sqrt)
    
    m,n = joint_samples.shape
    
    #Calculate the innovation sigma points from the joint sigma points
    Vi =  np.asmatrix(np.zeros((2,n)))
    
    for i in range(n):
        Vi[:,i] = CalcPtsInnovation(joint_samples[:,i])
    
    #Calculate the mean and covariance of the innovation
    V, Pvv = CalcMeanAndCovFromSigmaPts(Vi, joint_weights)
    
    #total sigma_pts of the state 
    Xi1 = joint_samples[0:2,:]

    n1 = mean1.size
    Pxv = np.asmatrix(np.zeros((n1,n1)))
    
    for k in range (n):
        s1 = (Xi1[:,k] - mean1)
        s2 = (Vi[:,k] - V)
        
        Pxv+= np.asscalar(joint_weights[k])*s1*s2.T
        
    K = - Pxv * (Pvv.I)
    
    #filter each sample
    fused_samples = Xi1 + K * Vi
    
    fused_state, fused_sqrt_cov = CalcMeanAndSqrtCovFromSigmaPts(fused_samples, joint_weights)
    
    return (fused_state, fused_sqrt_cov)    
    
    
############################## Generic SCI-SSIUKF ############################
NUM_CORRELATED_ERRORS = 3
NUM_MEAS_ERRORS = 1
scale_alpha = 0.05

class sqrt_cov_matrices_gen:
    def __init__(self,n):
        self.Q_sqrt_mat = numpy.asmatrix(numpy.zeros((n,n)))
        self.R0_sqrt_mat = numpy.asmatrix(numpy.zeros((n,n)))
        self.R1_sqrt_mat = numpy.asmatrix(numpy.zeros((n,n)))
        self.R2_sqrt_mat = numpy.asmatrix(numpy.zeros((n,n)))
        self.R3_sqrt_mat = numpy.asmatrix(numpy.zeros((n,n)))
        self.R4_sqrt_mat = numpy.asmatrix(numpy.zeros((n,n)))
        
        
    def get_pmat(self):
        return (supp.matrix_square(self.Q_sqrt_mat) + 
        supp.matrix_square(self.R0_sqrt_mat) + 
        supp.matrix_square(self.R1_sqrt_mat) + 
        supp.matrix_square(self.R2_sqrt_mat) + 
        supp.matrix_square(self.R3_sqrt_mat) + 
        supp.matrix_square(self.R4_sqrt_mat) )
    
def StatesAndMeasSigmasToZeroMeanGen(Q1,R10,R11,R12,R13,R14,Q2,R20,kappa=0.0):

    if numpy.isscalar(Q1):
        Q1 = numpy.matrix([[Q1]])
    if numpy.isscalar(R10):
        R10 = numpy.matrix([[R10]])
    if numpy.isscalar(R11):
        R11 = numpy.matrix([[R11]])
    if numpy.isscalar(R12):
        R12 = numpy.matrix([[R12]])
    if numpy.isscalar(R13):
        R13 = numpy.matrix([[R13]])
    if numpy.isscalar(R14):
        R14 = numpy.matrix([[R14]])
    if numpy.isscalar(Q2):
        Q2 = numpy.matrix([[Q2]])
    if numpy.isscalar(R20):
        R20 = numpy.matrix([[R20]])
        
    
    n_state = Q1.shape[0]
    n_meas = Q2.shape[0]
    
    n= n_state * (NUM_CORRELATED_ERRORS+1) + (NUM_MEAS_ERRORS+1) * n_meas
    n_state_samples= n_state * (NUM_CORRELATED_ERRORS+1) 
    n_meas_samples= (NUM_MEAS_ERRORS+1) * n_meas
    n_samples= 2 * n +1 
    
    W = numpy.asmatrix(numpy.full((n_samples, 1), 0.5/(n+kappa)))
    W[0] = kappa/(n+kappa)
    
    state_sample= numpy.matrix(numpy.zeros((n_state,n_state_samples)))
    meas_sample= numpy.matrix(numpy.zeros((n_meas,n_meas_samples)))
    
    state_sample[:,0:n_state]= Q1
    
    if(NUM_CORRELATED_ERRORS >= 1):
        state_sample[:,n_state:2*n_state]= R10
    if(NUM_CORRELATED_ERRORS >= 2):    
        state_sample[:,2*n_state:3*n_state]= R11
    if(NUM_CORRELATED_ERRORS >= 3):
        state_sample[:,3*n_state:4*n_state]= R12
    if(NUM_CORRELATED_ERRORS >= 4):
        state_sample[:,4*n_state:5*n_state]= R13
    if(NUM_CORRELATED_ERRORS >= 5):
        state_sample[:,5*n_state:6*n_state]= R14

    
    meas_sample[:,0:n_meas]= Q2
    if(NUM_MEAS_ERRORS>=1):
        meas_sample[:,n_meas:2*n_meas]= R20

    
    gamma= numpy.sqrt(n+kappa)
    state_sample= gamma * state_sample
    meas_sample= gamma * meas_sample
        
    return (state_sample,meas_sample, W)
    
    
def PointToPointInnovationUKF(state, meas, uls_focal1= numpy.matrix([[0.0],[0.0]]), uls_focal2= numpy.matrix([[0.0],[0.0]]) ):
    
    innovation = meas - state[0:2]
    
    return innovation
    
def PointEllipseInnovationUKF(state,meas,uls_focal1,uls_focal2):
    
    dist_x_f1 = supp.GetDistBtwPts(state[0:2,:],uls_focal1)
    dist_x_f2 = supp.GetDistBtwPts(state[0:2,:],uls_focal2)
    v = -meas + ((dist_x_f1 + dist_x_f2 )/2)

    return v
    
def PointAngleInnovationUKF(state,meas, uls_focal1= numpy.matrix([[0.0],[0.0]]), uls_focal2= numpy.matrix([[0.0],[0.0]]) ):
   
    state_angle= math.atan2(state[0,0],state[1,0])
    meas_angle= math.atan2(meas[0,0],meas[1,0])
    
    v = numpy.matrix([[(meas_angle - state_angle)]])

    return v
    
def SciSiUKFMerge(state1,state_sqrt_cov1,meas,meas_sqrt_cov ,sensor_type ,omega, UPDATE_FLAG = True, LIKELIHOOD_FLAG = True , MEASUREMENT_TYPE = 0, uls_focal1= numpy.matrix([[0.0],[0.0]]), uls_focal2= numpy.matrix([[0.0],[0.0]]) ):
        
    n_state = state1.size
    n_meas = meas.size
        
    alpha = 1.0 / omega
    beta = 1.0 / (1.0-omega)
    
    alpha_sqrt = numpy.sqrt(alpha)
    beta_sqrt = numpy.sqrt(beta)
    
    Q1_sqrt_op= state_sqrt_cov1.Q_sqrt_mat
    
    Q2_sqrt_op= meas_sqrt_cov.Q_sqrt_mat
    R20_sqrt_op= beta_sqrt * meas_sqrt_cov.R0_sqrt_mat
        
    if(sensor_type == 0):
        R10_sqrt_op=  alpha_sqrt * state_sqrt_cov1.R0_sqrt_mat
    else:
        R10_sqrt_op= state_sqrt_cov1.R0_sqrt_mat
    if(sensor_type == 1):
        R11_sqrt_op=  alpha_sqrt * state_sqrt_cov1.R1_sqrt_mat
    else:
        R11_sqrt_op= state_sqrt_cov1.R1_sqrt_mat
    if(sensor_type == 2):
        R12_sqrt_op=  alpha_sqrt * state_sqrt_cov1.R2_sqrt_mat 
    else:
        R12_sqrt_op=  state_sqrt_cov1.R2_sqrt_mat 
    if(sensor_type == 3):
        R13_sqrt_op=  alpha_sqrt * state_sqrt_cov1.R3_sqrt_mat 
    else:
        R13_sqrt_op=  state_sqrt_cov1.R3_sqrt_mat 
    if(sensor_type == 4):
        R14_sqrt_op=  alpha_sqrt * state_sqrt_cov1.R4_sqrt_mat 
    else:
        R14_sqrt_op=  state_sqrt_cov1.R4_sqrt_mat 
    
    """
    function StatesAndMeasSigmasToZeroMean --> returns two matrices 
    state_samples: matrix (n_state * (n_state * 4)) ::: 4 -> Q1 R10 R11 R12
            -> only the positive added part to mean (sqrt(n+kappa)*S) 
    meas_samples: matrix (n_meas * (n_meas * 2 ))   ::: 2 -> Q2 R20 
            -> only the positive added part to mean (sqrt(n+kappa)*S)
            
    ** This function returns the positive difference in sigma points to a zero mean (sqrt(n+kappa)*S)
        which minimizes the number of returned samples to (n) instead of (2*n+1)
    
    ** In addition, this function returns the State sample(4*16) and Measurement samples(2*4) directly
        rather than returns the joint state sample matrix(20*41)  
 
    Original samples ghraph
    =================================================================================================================
    =   State Samples due to(Q1 -Q1 R10 -R10 R11 -R11 R12 -R12)   |          State Mean (fixed & repeated)          =
    =-------------------------------------------------------------|-------------------------------------------------=
    =             Measurement Mean (fixed & repeated)             |    Measurement Samples due to(Q2 Q2 R20 -R20)   =
    =================================================================================================================
    
    Returned Matrices 
    =================================================================================================================
    =        [[  State Samples due to (Q1 R10 R11 R12)  ]]        |                                                 =
    =-------------------------------------------------------------|-------------------------------------------------=
    =                                                             |   [[ Measurement Samples due to (Q2 Q2 R20) ]]  =
    =================================================================================================================
    """
    state_samples,meas_samples,wi= StatesAndMeasSigmasToZeroMeanGen(Q1_sqrt_op,R10_sqrt_op,R11_sqrt_op,R12_sqrt_op,R13_sqrt_op,R14_sqrt_op,Q2_sqrt_op,R20_sqrt_op)  
    
    state_mean= state1
    meas_mean= meas
    
    n_complete_samples= wi.size
    n_state_sample= state_samples.shape[1]
    n_meas_sample= meas_samples.shape[1]
    
    if( (MEASUREMENT_TYPE == 0) and (n_meas == 2) ):
        n_innovation = 2
        h = PointToPointInnovationUKF
        
    elif( (MEASUREMENT_TYPE == 1) and (n_meas == 1) ):
        n_innovation = 1
        h= PointEllipseInnovationUKF
    
    elif( (MEASUREMENT_TYPE == 2) and (n_meas == 2) ):
        n_innovation = 1
        h= PointAngleInnovationUKF   

    vi = numpy.asmatrix(numpy.zeros((n_innovation, n_complete_samples)))
    #calculate the first innovation sample
    vi[:,0] = h(state_mean, meas_mean,uls_focal1,uls_focal2)
    
    """
        Complete the innovation samples using 2 for loop 
        one from 0 to n_state_sample with fixed measurement sample (meas_mean)
            --> saves time in scaling and unscaling fixed mean samples 
        and one from 0 to n_meas_sample with fixed state sample (state_mean) 
    """
    for j in range (n_state_sample):
        
        # Scale The state samples
        scaled_state_sample1 = state_mean + scale_alpha * state_samples[:,j]
        scaled_state_sample2 = state_mean - scale_alpha * state_samples[:,j]
        
        # calculate the scaled innovation sample for the part of state samples
        vi[:,2*j+1] = h(scaled_state_sample1, meas_mean,uls_focal1,uls_focal2)
        vi[:,2*j+2] = h(scaled_state_sample2, meas_mean,uls_focal1,uls_focal2)
        
        # Unscale the innovation sample
        vi[:,2*j+1] = ((vi[:,2*j+1] - vi[:,0]) / scale_alpha**2) + vi[:,0]
        vi[:,2*j+2] = ((vi[:,2*j+2] - vi[:,0]) / scale_alpha**2) + vi[:,0]
    
    for j in range (n_meas_sample):
        # Scale measurement samplses 
        scaled_meas_sample1 = meas_mean + scale_alpha * meas_samples[:,j]
        scaled_meas_sample2 = meas_mean - scale_alpha * meas_samples[:,j]
        
        # calculate the scaled innovation sample with fixed state mean sample
        vi[:,2*j+1+2*n_state_sample] = h(state_mean, scaled_meas_sample1, uls_focal1,uls_focal2)
        vi[:,2*j+2+2*n_state_sample] = h(state_mean, scaled_meas_sample2, uls_focal1,uls_focal2)
        
        # Unscale the innovation sample
        vi[:,2*j+1+2*n_state_sample] = ((vi[:,2*j+1+2*n_state_sample] - vi[:,0]) / scale_alpha**2) + vi[:,0]
        vi[:,2*j+2+2*n_state_sample] = ((vi[:,2*j+2+2*n_state_sample] - vi[:,0]) / scale_alpha**2) + vi[:,0]
    
    # Unscale the state difference samples 
    #state_samples= state_samples/self.scale_alpha

    #Calculate the mean and covariance of the innovation
    v, Pvv = CalcMeanAndCovFromSigmaPts(vi, wi)
    
    Pvv = (scale_alpha**2) * (Pvv)
    
    Pxv = numpy.asmatrix(numpy.zeros((n_state, n_innovation)))
    
    #calculate Pxv only from the part in which the expression (xi[j]-mean) not equals zero 
    for j in range (n_state_sample):
        
        Pxv+= numpy.asscalar(wi[2*j+1]) * (state_samples[:,j]) * (vi[:,2*j+1] - v).T 
        Pxv+= numpy.asscalar(wi[2*j+2]) * (-1 * state_samples[:,j]) * (vi[:,2*j+2] - v).T 
        
    Pxv = (scale_alpha) *(Pxv)
           
    k = -Pxv * Pvv.I
       
    Likelihood = 0
   
    if (LIKELIHOOD_FLAG == True):
        v_mean = numpy.matrix(numpy.zeros(v.shape))
        #Likelihood in the innovative KF is represented by the innovation space 
        #It represents how different is the calculated innovation from the expected zero innovation
        Likelihood = supp.CalcPdf(v, v_mean, Pvv)
        
#            self.maha_dist = v.T * Pvv.I * v
    
    #filter each sample
    fused_samples = numpy.asmatrix(numpy.zeros((n_state, n_complete_samples)))
    fused_samples[:,0] = state_mean + k * vi[:,0]
    
    for j in range(0,n_state_sample):
        fused_samples[:,2*j+1]= state_mean + state_samples[:,j]/scale_alpha + k * vi[:,2*j+1]
        fused_samples[:,2*j+2]= state_mean - state_samples[:,j]/scale_alpha + k * vi[:,2*j+2]
    
    for j in range(2 * n_meas_sample):
        fused_samples[:,j+2*n_state_sample+1]= state_mean + k * vi[:,j+2*n_state_sample+1]
         
    fused_mean= (state1 + k * v)
#        fused_mean = fused_samples * (wi)
    
    fused_sqrt_cov = sqrt_cov_matrices_gen(n_state)
    
    wi = (scale_alpha**2)*wi
    
    R20_START_INDEX = 2*n_meas+1 + 2*n_state_sample
    R20_END_INDEX = 4*n_meas+1 + 2*n_state_sample
       
    

    Q_sqrt_out = GetSplitSqrtCovFromSamplesAndMeanSci(fused_mean, fused_samples,wi, 1, 2*n_state+1, 
                                                                             2*n_state_sample+1, 2*n_meas+1 + 2*n_state_sample)
    if(NUM_CORRELATED_ERRORS >= 1):           
        if(sensor_type == 0):                                                              
            R0_sqrt_out = GetSplitSqrtCovFromSamplesAndMeanSci(fused_mean, fused_samples,wi, 
                                                                                   2*n_state+1, 4*n_state+1,
                                                                                   R20_START_INDEX, R20_END_INDEX)
                                                                          
        else:
            R0_sqrt_out = GetSplitSqrtCovFromSamplesAndMeanSci(fused_mean, fused_samples,wi, 
                                                                                   2*n_state+1, 4*n_state+1)
    if(NUM_CORRELATED_ERRORS >= 2):                                                                                   
        if(sensor_type == 1):
            R1_sqrt_out = GetSplitSqrtCovFromSamplesAndMeanSci(fused_mean, fused_samples,wi, 
                                                                                   4*n_state+1, 6*n_state+1,
                                                                                   R20_START_INDEX, R20_END_INDEX)
        else:
            R1_sqrt_out = GetSplitSqrtCovFromSamplesAndMeanSci(fused_mean, fused_samples,wi, 
                                                                                   4*n_state+1, 6*n_state+1)
    if(NUM_CORRELATED_ERRORS >= 3):                                                                                
        if(sensor_type == 2):
            R2_sqrt_out = GetSplitSqrtCovFromSamplesAndMeanSci(fused_mean, fused_samples,wi, 
                                                                                   6*n_state+1, 8*n_state+1,
                                                                                    R20_START_INDEX, R20_END_INDEX)
        else:
            R2_sqrt_out = GetSplitSqrtCovFromSamplesAndMeanSci(fused_mean, fused_samples,wi, 
                                                                                   6*n_state+1, 8*n_state+1)
    if(NUM_CORRELATED_ERRORS >= 4):                                                                               
        if(sensor_type == 3):
            R3_sqrt_out = GetSplitSqrtCovFromSamplesAndMeanSci(fused_mean, fused_samples,wi, 
                                                                                   8*n_state+1, 10*n_state+1,
                                                                                   R20_START_INDEX, R20_END_INDEX)
        else:
            R3_sqrt_out = GetSplitSqrtCovFromSamplesAndMeanSci(fused_mean, fused_samples,wi, 
                                                                               8*n_state+1, 10*n_state+1)           
    if(NUM_CORRELATED_ERRORS >= 5):                                                                           
        if(sensor_type == 4):
            R4_sqrt_out = GetSplitSqrtCovFromSamplesAndMeanSci(fused_mean, fused_samples,wi, 
                                                                                   10*n_state+1, 12*n_state+1,
                                                                                   R20_START_INDEX, R20_END_INDEX)
        else:
            R4_sqrt_out = GetSplitSqrtCovFromSamplesAndMeanSci(fused_mean, fused_samples,wi, 
                                                                               10*n_state+1, 12*n_state+1)                                                                                  
    
    fused_sqrt_cov.Q_sqrt_mat = Q_sqrt_out
    
    
    zero_matrix= numpy.asmatrix(numpy.zeros(( n_state, n_state)))
    if(NUM_CORRELATED_ERRORS == 0):
        fused_sqrt_cov.R0_sqrt_mat = zero_matrix
        fused_sqrt_cov.R1_sqrt_mat = zero_matrix
        fused_sqrt_cov.R2_sqrt_mat = zero_matrix 
        fused_sqrt_cov.R3_sqrt_mat = zero_matrix 
        fused_sqrt_cov.R4_sqrt_mat = zero_matrix
    if(NUM_CORRELATED_ERRORS == 1):
        fused_sqrt_cov.R0_sqrt_mat = R0_sqrt_out
        fused_sqrt_cov.R1_sqrt_mat = zero_matrix
        fused_sqrt_cov.R2_sqrt_mat = zero_matrix 
        fused_sqrt_cov.R3_sqrt_mat = zero_matrix 
        fused_sqrt_cov.R4_sqrt_mat = zero_matrix
    elif(NUM_CORRELATED_ERRORS == 2):
        fused_sqrt_cov.R0_sqrt_mat = R0_sqrt_out
        fused_sqrt_cov.R1_sqrt_mat = R1_sqrt_out
        fused_sqrt_cov.R2_sqrt_mat = zero_matrix 
        fused_sqrt_cov.R3_sqrt_mat = zero_matrix 
        fused_sqrt_cov.R4_sqrt_mat = zero_matrix
    elif(NUM_CORRELATED_ERRORS == 3):
        fused_sqrt_cov.R0_sqrt_mat = R0_sqrt_out
        fused_sqrt_cov.R1_sqrt_mat = R1_sqrt_out
        fused_sqrt_cov.R2_sqrt_mat = R2_sqrt_out
        fused_sqrt_cov.R3_sqrt_mat = zero_matrix 
        fused_sqrt_cov.R4_sqrt_mat = zero_matrix
    elif(NUM_CORRELATED_ERRORS == 4):
        fused_sqrt_cov.R0_sqrt_mat = R0_sqrt_out
        fused_sqrt_cov.R1_sqrt_mat = R1_sqrt_out
        fused_sqrt_cov.R2_sqrt_mat = R2_sqrt_out
        fused_sqrt_cov.R3_sqrt_mat = R3_sqrt_out
        fused_sqrt_cov.R4_sqrt_mat = zero_matrix
    elif(NUM_CORRELATED_ERRORS == 5):
        fused_sqrt_cov.R0_sqrt_mat = R0_sqrt_out
        fused_sqrt_cov.R1_sqrt_mat = R1_sqrt_out
        fused_sqrt_cov.R2_sqrt_mat = R2_sqrt_out
        fused_sqrt_cov.R3_sqrt_mat = R3_sqrt_out
        fused_sqrt_cov.R4_sqrt_mat = R4_sqrt_out
    
    
    det_Pn = 0.0
    fused_output_sqrt_cov = sqrt_cov_matrices_gen(n_state)
    fused_output_mean = numpy.matrix(numpy.zeros((n_state,1)))
    
    if (UPDATE_FLAG == True):
        fused_output_mean = fused_mean
        fused_output_sqrt_cov = fused_sqrt_cov
    else:
        det_Pn = numpy.linalg.det(fused_sqrt_cov.get_pmat())

   
    return fused_output_mean,fused_output_sqrt_cov,Likelihood,det_Pn        

################################### DEBUGGING ################################
    
################################### Input ####################################
# Noise in state
#Pxx = matrix([[9000.0,0.0],[0.0,6000.0]])
#
## Noise in the driven distance and wheel angle
#Puu = matrix([[200.0,0.0],[0.0,10.0]])
#
#state = matrix([[150.0],[100.0]])
#Input = matrix([[100.0], [45.0]])
#
#cov = supp.JoinMatsOnDiag(Pxx, Puu)
#
#mean = matrix([[state[0,0]],[state[1,0]],[Input[0,0]],[Input[1,0]]])
#kappa = 0.0
#
#x1 = matrix([[100.0],[300.0],[10.0],[7.0]])
#x2 = matrix([[150.0],[200.0]])
#c1 = matrix ([[2000.0,0.0,12.,0.0],[0.0,1000.0,0.0,10.0],[12.,0.0,20.,0.0],[0.0,10.,0.0,50.]])
#c2 = matrix ([[300.0,0.0],[0.0,150.0]])
##
#predicted_samples =  matrix([[1.000000000000000000e+02,	3.000000000000000000e+02],
#                            [1.632455532033675922e+02,	3.000000000000000000e+02],
#                            [1.000000000000000000e+02,	3.447213595499957819e+02],
#                            [3.675444679663241487e+01,	3.000000000000000000e+02],
#                            [1.000000000000000000e+02,	2.552786404500042181e+02]])
#                            
#predicted_samples_weights = matrix ([[0.000000000000000000e+00],
#                                    [2.500000000000000000e-01],
#                                    [2.500000000000000000e-01],
#                                    [2.500000000000000000e-01],
#                                    [2.500000000000000000e-01]])
############################### Unscented Kalman Filter #######################

# Prediction Step 

#
#mean1, cov1 = ukf_predict(cov, mean)
#
#cov1_sqrt = linalg.cholesky(cov1)
#cov_sqrt = linalg.cholesky(cov)
#mean2, cov_sqrt2,xi2,w2 = iesukf_predict(cov_sqrt,mean)
#cov2 = cov_sqrt2 * cov_sqrt2.T
#
#Xn, Pn = ukf_merge(x1, c1, x2, c2)
#
#Xn2, Pn2 = scaled_ukf_merge(x1, c1, x2, c2)
#
#print Xn - Xn2
#print Pn - Pn2

#
#Xn, Pn_sqrt = iesukf_merge(x1, c1, predicted_samples.T, predicted_samples_weights, x2, c2)
#Pn = Pn_sqrt * Pn_sqrt.T
#
#Xn1,Pn1 = linear_merge(x1, c1, x2, c2)
#
#print 'diff:'
#print Xn-Xn1
#print Pn-Pn1

################################## PLOT FUNCTIONS #############################
#fig = plt.figure('x-y plane')
#plt.axis([50, 250, 50, 200])
#plt.grid(True)
#ax = fig.add_subplot(1,1,1)
#fig.set_size_inches(10.5, 10.5)
#ax.set_aspect('equal')

##mean1_unrot = np.asmatrix(np.zeros((2,1)))
##rot_trans(mean1, Input[0,0], Input[1,0] , mean1_unrot)
##
##cov1_unrot = supp.rot_cov_mat(cov1, Input[1,0])
#
#mean2_unrot = np.asmatrix(np.zeros((2,1)))
#rot_trans(mean2, Input[0,0], Input[1,0] , mean2_unrot)
#
#cov2_unrot = supp.rot_cov_mat(cov2, Input[1,0])
#
##supp.DrawPoseWithEllipse(ax, mean1_unrot, cov1_unrot, 'r')
##supp.DrawPoseWithEllipse(ax, mean, cov, 'b')
#
#supp.DrawPoseWithEllipse(ax, mean2_unrot, cov2_unrot, 'r')
#supp.DrawPoseWithEllipse(ax, mean, cov, 'b')

#
#supp.DrawPoseWithEllipse(ax, x1[0:2], c1[0:2,0:2], 'r','r')
#supp.DrawPoseWithEllipse(ax, x2, c2, 'b','b')
#supp.DrawPoseWithEllipse(ax, Xn[0:2], Pn[0:2], 'k','k')
