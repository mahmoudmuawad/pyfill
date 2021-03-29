import numpy as np
from math import cos,sin,atan2,exp,pi,sqrt

alpha1=0.1
alpha2=0.12
alpha3=0.25
alpha4=0.2

def Odom_model(xt, ut, xt_1, prob) :
    xt_bar=ut[0,0]         # where xt_bar = [x_bar_dash y_bar_dash theta_bar_dash]
    xt_1_bar=ut[1,0]       # where xt_1_bar = [x_bar y_bar theta_bar]
    
    dRot1=atan2(xt_bar[1]-xt_1_bar[1],xt_bar[0]-xt_1_bar[0])-xt_1_bar[2]
    dTrans=sqrt((xt_1_bar[0]-xt_bar[0])**2 + (xt_1_bar[1]-xt_bar[1])**2)
    dRot2=xt_bar[2]-xt_1_bar[2]-dRot1
    
    dRot1_hat=atan2(xt[1]-xt_1[1],xt[0]-xt_1[0])-xt_1[2]
    dTrans_hat=sqrt((xt_1[0]-xt[0])**2 + (xt_1[1]-xt[1])**2)
    dRot2_hat=xt[2]-xt_1[2]-dRot1_hat
    
    p1=prob(dRot1-dRot1_hat, alpha1*dRot1_hat+alpha2*dTrans_hat)
    p2=prob(dTrans-dTrans_hat, alpha3*dTrans_hat+alpha4*(dRot1_hat+dRot2_hat))
    p3=prob(dRot2-dRot2_hat, alpha1*dRot2_hat+alpha2*dTrans_hat)
    
    return p1*p2*p3

def normal_prob(a, b):
    return 1/sqrt(2*pi*b) * exp(-0.5*a**2/b)

def tri_prob(a,b):
    if abs(a)>sqrt(6*b) :
        return 0
    else :
        return (sqrt(6*b)-abs(a))/(6*b)
    
def sample_Odom_model(ut, xt_1, sample):
    xt_bar=ut[0,0]         # where xt_bar = [x_bar_dash y_bar_dash theta_bar_dash]
    xt_1_bar=ut[1,0]       # where xt_1_bar = [x_bar y_bar theta_bar]
    
    dRot1=atan2(xt_bar[1]-xt_1_bar[1],xt_bar[0]-xt_1_bar[0])-xt_1_bar[2]
    dTrans=sqrt((xt_1_bar[0]-xt_bar[0])**2 + (xt_1_bar[1]-xt_bar[1])**2)
    dRot2=xt_bar[2]-xt_1_bar[2]-dRot1
    
    dRot1_hat=dRot1-sample(alpha1*dRot1+alpha2*dTrans)
    dTrans_hat=dTrans-sample(alpha3*dTrans+alpha4*(dRot1+dRot2))
    dRot2_hat=dRot2-sample(alpha1*dRot2+alpha2*dTrans)
    
    xt=np.matrix(np.zeros((3,1)))
    xt[0,0]=xt_1[0]+dTrans_hat*cos(xt_1[2]+dRot1_hat)
    xt[1,0]=xt_1[1]+dTrans_hat*sin(xt_1[2]+dRot1_hat)
    xt[2,0]=xt_1[2]+dRot1_hat+dRot2_hat
    
    return xt

def normal_sample(b):    
    ret=0
    for i in range(1,13):
        ret+=np.random.uniform(-1,1)
    return ret*b/6

def tri_sample(b):
    return b*np.random.uniform(-1,1)*np.random.uniform(-1,1)
    
################# test ###############################    