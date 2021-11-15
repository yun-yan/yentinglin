#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:21:45 2021

@author: yentinglin
"""


import numpy as np
from numpy import pi
from numba import njit#,float64,complex128#,jit_integrand_function
from scipy.integrate import solve_ivp#,quad
from numpy.linalg import inv
from tool.tool_solver_eq import frg_eq_irlm
#from functools import partial
#import multiprocessing
import cmath


@njit()
def sigma_retarded_bath(w,t1,t2,v,mu):

    delta=1e-7
    #delta_edge=1e-20

    d=np.sqrt( np.power(v,2.0)+np.power((t1-t2),2.0) ) 
    dd=np.sqrt( np.power(v,2.0)+np.power((t1+t2),2.0) ) 
    sigma_bath=0.0+1.0j*0.0
    
    z=w-mu+1.0j*delta
    sigma_bath=(np.power(z,2.0)-np.power(v,2.0)-np.power(t2,2.0)+np.power(t1,2.0)-cmath.sqrt(z-d)*cmath.sqrt(z+d)*cmath.sqrt(z-dd)*cmath.sqrt(z+dd)   )  /(2.0*(z-v))

    return sigma_bath

@njit()
def g_retardation(w,beta,e,t_l,t_r,
                  t1_l,t2_l,v_l,mu_l,
                  t1_r,t2_r,v_r,mu_r):
                  
    delta=1e-7
    #sigma_retarded_bath(w,t1,t2,v,mu)
                 
    sigma_bath_l=sigma_retarded_bath(w,t1_l,t2_l,v_l,mu_l)
    sigma_bath_r=sigma_retarded_bath(w,t1_r,t2_r,v_r,mu_r)

    inv_g=np.zeros((3,3),dtype=np.complex128) 
    #print(mu_l,mu_r)
    inv_g[0,0]=w+1.0j*delta-sigma_bath_l-mu_l#
    inv_g[1,1]=w+1.0j*delta-e
    inv_g[2,2]=w+1.0j*delta-sigma_bath_r-mu_r#
    
    inv_g[0,1]=t_l
    inv_g[1,0]=t_l
    inv_g[1,2]=t_r
    inv_g[2,1]=t_r

    g=inv(inv_g)

    return g

@njit()
def eff_g_retardation(sol,w,beta,e,t_l,t_r,
                      t1_l,t2_l,v_l,mu_l,
                      t1_r,t2_r,v_r,mu_r):
                  
    delta=1e-15
                 #sigma_retarded_bath(w,t1,t2,v,mu)
                 
    sigma_bath_l=sigma_retarded_bath(w,t1_l,t2_l,v_l,mu_l)
    sigma_bath_r=sigma_retarded_bath(w,t1_r,t2_r,v_r,mu_r)

    inv_g=np.zeros((3,3),dtype=np.complex128) 
    #print(mu_l,mu_r)
    inv_g[0,0]=w+1.0j*delta-sigma_bath_l-mu_l-sol[0]
    inv_g[1,1]=w+1.0j*delta-e-sol[1]
    inv_g[2,2]=w+1.0j*delta-sigma_bath_r-mu_r-sol[0]
    
    inv_g[0,1]=t_l-sol[2]
    inv_g[1,0]=t_l-np.conjugate(sol[2])
    inv_g[1,2]=t_r-sol[3]
    inv_g[2,1]=t_r-np.conjugate(sol[3])

    g=inv(inv_g)

    return g

@njit()
def eff_g_retardation_two_level(sol,w,beta,e,t_l,t_r,
                                t1_l,t2_l,v_l,mu_l,
                                t1_r,t2_r,v_r,mu_r):
                  
    delta=1e-4
                 #sigma_retarded_bath(w,t1,t2,v,mu)
                 
    sigma_bath_l=sigma_retarded_bath(w,t1_l,t2_l,v_l,mu_l)
    #sigma_bath_r=sigma_retarded_bath(w,t1_r,t2_r,v_r,mu_r)

    inv_g=np.zeros((2,2),dtype=np.complex128) 
    #print(mu_l,mu_r)
    inv_g[0,0]=w+1.0j*delta-sigma_bath_l-mu_l-sol[0]
    inv_g[1,1]=w+1.0j*delta-e-sol[1]
    #inv_g[2,2]=w+1.0j*delta-sigma_bath_r-mu_r-sol[0]
    
    inv_g[0,1]=t_l-sol[2]
    inv_g[1,0]=t_l-np.conjugate(sol[2])
    #inv_g[1,2]=t_r-sol[3]
    #inv_g[2,1]=t_r-np.conjugate(sol[3])

    g=inv(inv_g)

    return g

