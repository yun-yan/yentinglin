#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 09:38:58 2021

@author: yentinglin
"""

import numpy as np
from numpy import tanh,exp,pi,log
from numba import njit,float64,complex128#,jit_integrand_function
from scipy.integrate import solve_ivp,quad
from numpy.linalg import inv
from functools import partial
import multiprocessing
import cmath

@njit()
def sigma_retarded_bath_eq(gamma,l,dt,v=0.0,mu=0.0):

    delta=5e-10
    t1=(gamma+dt)/2.0
    t2=(gamma-dt)/2.0
    d=np.sqrt( np.power(v,2.0)+np.power(t1-t2,2.0 ) )
    dd=np.sqrt( np.power(v,2.0)+np.power(t1+t2,2.0 ) )
    sigma_bath=0.0+1.0j*0.0
    z=1.0j*l-mu+1.0j*delta
    #sigma_bath=(np.power(z,2.0)-np.power(v,2.0)-np.power(t2,2.0)+np.power(t1,2.0)-cmath.sqrt((z-d)*(z+d))*cmath.sqrt((z-dd)*(z+dd))   )  /(2.0*(z-v))
    sigma_bath=(np.power(z,2.0)-np.power(v,2.0)-np.power(t2,2.0)+np.power(t1,2.0)-cmath.sqrt(z-d)*cmath.sqrt(z+d)*cmath.sqrt(z-dd)*cmath.sqrt(z+dd)   )  /(2.0*(z-v))
    
    return sigma_bath


@njit()
def g_eq(y,l,gamma,t,e,beta,dt_l,dt_r,v_l,v_r,mu_l,mu_r):
                  
    delta=5e-10

    sigma_bath_l=sigma_retarded_bath_eq(gamma,l,dt_l,v_l,mu_l)
    sigma_bath_r=sigma_retarded_bath_eq(gamma,l,dt_r,v_r,mu_r)

    inv_g=np.zeros((3,3),dtype=np.complex128) 

    inv_g[0,0]=1.0j*(delta+l)-sigma_bath_l-mu_l-v_l-y[0]
    inv_g[1,1]=1.0j*(delta+l)-e-y[1]
    inv_g[2,2]=1.0j*(delta+l)-sigma_bath_r-mu_r-v_r-y[0]
    
    inv_g[0,1]=t-y[2]
    inv_g[1,0]=t-np.conjugate(y[2])
    inv_g[1,2]=t-y[3]
    inv_g[2,1]=t-np.conjugate(y[3])

    g=inv(inv_g)

    return g



def frg_eq_irlm(gamma,t,e,u,beta,dt_l,dt_r,v_l=0.0,v_r=0.0,mu_l=0.0,mu_r=0.0):
    
    def diffeq(l, y):
        ## y are solutions of differential equations: set
        ## l is the flow parameter of differential equations: float
        ## single scale propagator for positive cutoff
        #
        #print(l)

        g_ret=g_eq(y,l,gamma,t,e,beta,dt_l,dt_r,v_l,v_r,mu_l,mu_r)

         
        diffeq=np.zeros(4,dtype=np.complex128) 
        diffeq[0]=-(u*g_ret[1,1]).real/(np.pi)
        diffeq[1]=-u*(g_ret[0,0]+g_ret[2,2]).real/(np.pi)
        diffeq[2]=u*(g_ret[0,1]).real/(np.pi)
        diffeq[3]=u*(g_ret[1,2]).real/(np.pi)
        
        return diffeq 

    #CUTOFF_LIST=np.zeros(100,dtype=np.float64)
    #for i in range(100):
    #    a=1e4*np.power(0.703,i)
    #    CUTOFF_LIST[i]=a
    
    #initial condition of the solution
    init=np.zeros(4,dtype=np.complex128) 
    flow_IR=1e-13#t*t/10.0
    flow=[1e3,flow_IR]

    #relative accuracy in RK45 solver
    acc_rtol=1.48e-13
    acc_atol=1.48e-13
    sol=solve_ivp(diffeq,flow,init,method='RK45',rtol=acc_rtol,atol=acc_atol)
           
    return sol