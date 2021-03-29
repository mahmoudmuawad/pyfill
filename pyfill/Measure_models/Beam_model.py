import numpy as np
from math import *

#define range sensor parameters  
z_max=12
std_hit = 0.15
lamda_short=1.1
def p_hit(zt, xt, m, ray_casting, std_hit=std_hit):
    if(zt>0 and zt<z_max):
        zt_star=ray_casting(zt)
        eita_inv = -1/(std_hit**2*sqrt(2*pi*std_hit**2))*(exp(-0.5*(z_max-zt_star)**2/std_hit**2) *\
                       (z_max-zt_star) +exp(-0.5*(0-zt_star)**2/std_hit**2)*(z_max-zt_star))
        return 1/eita_inv * 1/sqrt(2*pi*std_hit)*exp(-0.5*(zt[k]-zt_star)**2/std_hit**2)
    else :
        return 0
    

def p_short(zt, xt, m, ray_casting, lamda_short=lamda_short):
    zt_star=ray_casting(zt)
    eita=1/(1-exp(-lamda_short*zt_star))
    if(zt>=0 and zt<=zt_star)
        return eita*lamda_short*exp(-lamda_short*zt)
    else:
        return 0

def p_max(zt, xt,m):
    if(zt==z_max):
        return 1
    return 0

def p_rand(zt, xt, m):
    if(zt >=0 and zt<z_max):
        return 1/z_max
    return 0
    
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

def beam_range_finder_model(zt, xt, m, ray_casting, intrinsi_params) :
    z_hit=intrinsi_params[0]
    z_short=intrinsi_params[1]
    z_max=intrinsi_params[2]
    z_rand=intrinsi_paramsp[3]
    std_hit=intrinsi_params[4]
    lamda_short=intrinsi_params[5]
    q=1
    for k in range(zt.shape[0]):
        zk_star=ray_casting(zt)
        p=z_hit*p_hit(zt[k], xt, m, ray_casting,  std_hit)+ \
        z_short*p_short(zt[k], xt, m, ray_casting, lamda_short)+z_max*p_max(zt[k],xt,m)+ \
        z_rand*p_rand(zt[k],xt,m)
        q*=p
    return q








