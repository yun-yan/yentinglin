#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 11:37:09 2021

@author: yentinglin
"""

import numpy as np
import sys 
sys.path.append('..')

from tool.tool_spectral_function import g_retardation,eff_g_retardation,eff_g_retardation_two_level
from tool.tool_solver_eq import frg_eq_irlm
import matplotlib.pyplot as plt

def spf(N,d,W,T,E,U,DT_L,DT_R,BETA=0.0):
    
    GAMMA=W#/2.0
    sol_frg_TN_u=frg_eq_irlm(GAMMA,T,E,U,BETA,DT_L,DT_R)


    SF_FREE=np.zeros([3,N],dtype=np.float64)
    SF_FRG=np.zeros([3,N],dtype=np.float64)
    LIST_OMEGA=np.zeros(N,dtype=np.float64)
    

    for i in range(N):

        omega=-0.3*d+0.6*d*i/N
        LIST_OMEGA[i]=omega
        T1_L=W/2.0-d/2.0
        T2_L=W/2.0+d/2.0
        #T1_R=T2_R=W/2.0
        T1_R=W/2.0#-d/2.0
        T2_R=W/2.0#+d/2.0
        
        #g_ret_TN=g_retardation(omega,BETA,E,T,T,
        #                       T1_L,T2_L,0.0,0.0,
        #                       T1_R,T2_R,0.0,0.0)
   
    
        g_ret_TN_U=eff_g_retardation_two_level(sol_frg_TN_u.y[:,-1],omega,BETA,E,T,T,
                                               T1_L,T2_L,0.0,0.0,
                                               T1_R,T2_R,0.0,0.0)
    
        for ii in range(2):
            #SF_FREE[ii,i]=-g_ret_TN[ii,ii].imag/np.pi
            SF_FRG[ii,i]=-g_ret_TN_U[ii,ii].imag/np.pi
    return [LIST_OMEGA,SF_FREE,SF_FRG]

def spf_prefactor(N,d,W,T,E,U,DT_L,DT_R,omega_1,omega_2,BETA=0.0):
    
    GAMMA=W#/2.0
    sol_frg_TN_u=frg_eq_irlm(GAMMA,T,E,U,BETA,DT_L,DT_R)


    #SF_FREE=np.zeros([3,N],dtype=np.float64)
    SF_FRG=np.zeros(N,dtype=np.float64)
    LIST_OMEGA=np.zeros(N,dtype=np.float64)
    

    for i in range(N):

        omega=omega_1+(omega_2-omega_1)*i/N
        LIST_OMEGA[i]=omega
        T1_L=W/2.0-d/2.0
        T2_L=W/2.0+d/2.0
        #T1_R=T2_R=W/2.0
        T1_R=W/2.0#-d/2.0
        T2_R=W/2.0#+d/2.0
        
        #g_ret_TN=g_retardation(omega,BETA,E,T,T,
        #                       T1_L,T2_L,0.0,0.0,
        #                       T1_R,T2_R,0.0,0.0)
   
    
        g_ret_TN_U=eff_g_retardation_two_level(sol_frg_TN_u.y[:,-1],omega,BETA,E,T,T,
                                               T1_L,T2_L,0.0,0.0,
                                               T1_R,T2_R,0.0,0.0)
    
        #for ii in range(2):
            #SF_FREE[ii,i]=-g_ret_TN[ii,ii].imag/np.pi
        SF_FRG[i]=-g_ret_TN_U[1,1].imag/np.pi
    return [LIST_OMEGA,SF_FRG]

def spf_prefactor_topological(N,d,W,T,E,U,DT_L,DT_R,omega_1,omega_2,BETA=0.0):
    
    GAMMA=W#/2.0
    sol_frg_TN_u=frg_eq_irlm(GAMMA,T,E,U,BETA,DT_L,DT_R)


    #SF_FREE=np.zeros([3,N],dtype=np.float64)
    SF_FRG=np.zeros(N,dtype=np.float64)
    LIST_OMEGA=np.zeros(N,dtype=np.float64)
    

    for i in range(N):

        omega=omega_1+(omega_2-omega_1)*i/N
        LIST_OMEGA[i]=omega
        T1_L=W/2.0+d/2.0
        T2_L=W/2.0-d/2.0
        #T1_R=T2_R=W/2.0
        T1_R=W/2.0#-d/2.0
        T2_R=W/2.0#+d/2.0
        
        #g_ret_TN=g_retardation(omega,BETA,E,T,T,
        #                       T1_L,T2_L,0.0,0.0,
        #                       T1_R,T2_R,0.0,0.0)
   
    
        g_ret_TN_U=eff_g_retardation_two_level(sol_frg_TN_u.y[:,-1],omega,BETA,E,T,T,
                                               T1_L,T2_L,0.0,0.0,
                                               T1_R,T2_R,0.0,0.0)
    
        #for ii in range(2):
            #SF_FREE[ii,i]=-g_ret_TN[ii,ii].imag/np.pi
        SF_FRG[i]=-g_ret_TN_U[1,1].imag/np.pi
    return [LIST_OMEGA,SF_FRG]

