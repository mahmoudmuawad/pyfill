from math import cos,sin,sqrt,atan2,pi,exp
import numpy as np
# define parameters
alpha1=0.1
alpha2=0.12
alpha3=0.25
alpha4=0.2
alpha5=0.13
alpha6=0.16

def motion_model_velocity(xt, ut, xt_1, prob):
    # xt = [x_dash y_dash theta_dash]  , xt_1 = [x y z], ut=[ut[0]w]
    mu=0.5*((xt_1[0]-xt[0])*cos(xt_1[2])+(xt_1[1]-xt[1])*sin(xt_1[2])) / \
    ((xt_1[1]-xt[1])*cos(xt_1[2])+(xt_1[0]-xt[0])*sin(xt_1[2]))
    
    xstar=(xt_1[0]+xt[0])/2+mu*(xt_1[1]-xt[1])
    ystar=(xt_1[1]+xt[1])/2+mu*(xt[0]-xt_1[0])
    rstar=sqrt((xt_1[0]-xstar)**2+(xt_1[1]-ystar)**2)
    
    dtheta=atan2(xt[1]-ystar, xt[0]-xstar)-atan2(xt_1[1]-ystar, xt_1[0]-xstar)
    
    v_hat=dtheta/dt*rstar
    w_hat=dtheta/dt
    gamma=(xt[2]-xt_1[2])/2-w_hat
    
    return prob(ut[0]-v_hat, alpha1*abs(ut[0])+alpha2*abs(ut[1]))*prob(ut[1]-w_hat, alpha3*abs(ut[0])+\
                alpha4*abs(ut[1]))*prob(gamma,alpha5*abs(ut[0])+alpha6*abs(ut[1]))
    
    
def normal_prob(a, b):
    return 1/sqrt(2*pi*b) * exp(-0.5*a**2/b)

def tri_prob(a,b):
    if abs(a)>sqrt(6*b) :
        return 0
    else :
        return (sqrt(6*b)-abs(a))/(6*b)
    
def sample_v_model(ut, xt_1, sample):
    v_hat=u[0]+sample(alpha1*abs(u[0])+alpha2*abs(u[1]))
    w_hat=u[1]+sample(alpha3*abs(u[0])+alpha4*abs(u[1]))
    gamma=sample(alpha5*abs(u[0])+alpha6*abs(u[1]))
    
    xt=np.matrix(np.zeros((3,1)))
    xt[0,0]=xt_1[0]-v_hat/w_hat*sin(xt_1[2])+v_hat/w_hat*sin(xt_1[2]+w_hat*dt)
    xt[1,0]=xt_1[1]+v_hat/w_hat*cos(xt_1[2])-v_hat/w_hat*cos(xt_1[2]+w_hat*dt)
    xt[2,0]=xt_1[2]+w_hat*dt+gamma*dt
    
    return xt


def normal_sample(b):
    ret=0
    for i in range(1,13):
        ret+=np.random.uniform(-1,1)
    return ret*b/6

def tri_sample(b):
    return b*np.random.uniform(-1,1)*np.random.uniform(-1,1)

################# test ###############################  


