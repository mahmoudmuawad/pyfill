from math import * 
import numpy as np

#define parameters
z_max=1.2
z_rand=0.5
std_hit=0.8
z_hit=0.02
x_sens=0.0
y_sens=0.05
theta_sens=0.0

def likelihood_range_finder_model(zt, xt, m, prob):
    q=1
    for k in range(zt.shape[0]):
        if(zt[k]!=z_max):
            xzt = xt[0]+x_sens*cos(xt[2])-y_sens*sin(xt[2])+zt[k]*cos(xt[2]+theta_sens)
            yzt = xt[1]+y_sens*cos(xt[2])+x_sens*sin(xt[2])+zt[k]*sin(xt[2]+theta_sens)
            dist_p2 = z_max**2
            for mk in m:
                if(((xzt-mk[0])**2 + (yzt-mk[1])**2) < dist_p2):
                    dist_p2=((xzt-mk[0])**2 + (yzt-mk[1])**2)
            q*=z_hit*prob(0,std_hit, dist_p2)+z_rand/z_max
    return q

def Gaussian_prob(mu, sigma, x):
    # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
    return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))

def learn_intrinsic_params(Z, X, m, ray_casting):
    ei_hit_sum=0
    ei_short_sum=0
    ei_max_sum=0
    ei_rand_sum=0
    ei_hit_sum_diff=0
    for i in range(Z.shape[0]):
        eita=1/(p_hit(Z[k],X[k], m, ray_casting)+p_short(Z[k], X[k],m, ray_casting)+ \
                p_max(Z[k], X[k],m)+p_rand(Z[k], X[k],m))
        zt_star=ray_casting(Z[k])
        ei_hit_sum+=eita*p_hit(Z[k],X[k], m, ray_casting)
        ei_hit_sum_diff+=eita*p_hit(Z[k],X[k], m, ray_casting)*(Z[k]-zt_star)**2
        ei_short_sum+=eita*p_short(Z[k], X[k],m, ray_casting)
        ei_short_sum_z=eita*p_short(Z[k], X[k],m, ray_casting)*Z[k]
        ei_max_sum+=eita*p_max(Z[k], X[k],m)
        ei_rand_sum+=eita*p_rand(Z[k], X[k],m)
        
    normalizer=Z.shape[0]
    z_hit=1/normalizer*ei_hit_sum
    z_short=1/normalizer*ei_short_sum
    z_max=1/normalizer*ei_max_sum
    z_rand=1/normalizer*ei_rand_sum
    std_hit=sqrt(1/ei_hit_sum*ei_hit_sum_diff)
    lamda_short=ei_short_sum/ei_short_sum_z

    return [z_hit,z_short,z_max,z_rand,std_hit,lamda_short]