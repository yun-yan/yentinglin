#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 11:35:03 2021

@author: yentinglin
"""


import numpy as np
import sys 
from tool_dsf import spf,spf_prefactor,spf_prefactor_topological
#from tool.tool_spectral_function import g_retardation,eff_g_retardation
from tool_solver_eq import frg_eq_irlm
import matplotlib.pyplot as plt
#from tool.tool_solver_eq import frg_eq_irlm


#N_b=100000000000000
BETA=np.inf
W=1.0



T=5.0e-3#0.1*d#np.sqrt(G*W/(4.0*np.pi))
G=8.0*T*T/W

d=100.0*G

U=0.22


E=0.2*d#0.15*d
DT_L=-d
DT_R=0.0


N=400000

GAMMA=W#/2.0

#%%
r=5.0
u1=0.0#01*G
u2=25*G#u1/r#/2.0
u3=50*G#u2/r#/2.0
u4=75*G#u3/r#/2.0
u5=100*G#u4/r#/2.0
u=0.

T1=T
T2=T1#/r
T3=T2#/r
T4=T3#/r
T5=T4#/r
a1=spf(N,d,W,T1,E,u1,DT_L,DT_R,BETA=0.0)
a2=spf(N,d,W,T2,E,u2,DT_L,DT_R,BETA=0.0)
a3=spf(N,d,W,T3,E,u3,DT_L,DT_R,BETA=0.0)
a4=spf(N,d,W,T4,E,u4,DT_L,DT_R,BETA=0.0)
a5=spf(N,d,W,T5,E,u5,DT_L,DT_R,BETA=0.0)

b1=spf_prefactor(N,d,W,T1,E,u1,DT_L,DT_R,-0.1*d,0.0,BETA=0.0)
b2=spf_prefactor(N,d,W,T2,E,u2,DT_L,DT_R,-0.1*d,0.0,BETA=0.0)
b3=spf_prefactor(N,d,W,T3,E,u3,DT_L,DT_R,-0.1*d,0.0,BETA=0.0)
b4=spf_prefactor(N,d,W,T4,E,u4,DT_L,DT_R,-0.1*d,0.0,BETA=0.0)
b5=spf_prefactor(N,d,W,T5,E,u5,DT_L,DT_R,-0.1*d,0.0,BETA=0.0)


c1=spf_prefactor(N,d,W,T1,E,u1,DT_L,DT_R,0.17*d,0.3*d,BETA=0.0)
c2=spf_prefactor(N,d,W,T2,E,u2,DT_L,DT_R,0.17*d,0.3*d,BETA=0.0)
c3=spf_prefactor(N,d,W,T3,E,u3,DT_L,DT_R,0.17*d,0.3*d,BETA=0.0)
c4=spf_prefactor(N,d,W,T4,E,u4,DT_L,DT_R,0.17*d,0.3*d,BETA=0.0)
c5=spf_prefactor(N,d,W,T5,E,u5,DT_L,DT_R,0.17*d,0.3*d,BETA=0.0)

#%%
e1=spf_prefactor_topological(N,d,W,T1,E,u1,-DT_L,DT_R,0.8*E,1.2*E,BETA=0.0)
e2=spf_prefactor_topological(N,d,W,T2,E,u2,-DT_L,DT_R,0.8*E,1.2*E,BETA=0.0)
e3=spf_prefactor_topological(N,d,W,T3,E,u3,-DT_L,DT_R,0.8*E,1.2*E,BETA=0.0)
e4=spf_prefactor_topological(N,d,W,T4,E,u4,-DT_L,DT_R,0.8*E,1.2*E,BETA=0.0)
e5=spf_prefactor_topological(N,d,W,T5,E,u5,-DT_L,DT_R,0.8*E,1.2*E,BETA=0.0)


#%%

delta_1=0.1*d
delta_2=0.13*d
delta_3=0.4*E



#%%
peak1_a=np.sum(b1[1])/N
peak2_a=np.sum(b2[1])/N
peak3_a=np.sum(b3[1])/N
peak4_a=np.sum(b4[1])/N
peak5_a=np.sum(b5[1])/N

peak1_b=np.sum(c1[1])/N
peak2_b=np.sum(c2[1])/N
peak3_b=np.sum(c3[1])/N
peak4_b=np.sum(c4[1])/N
peak5_b=np.sum(c5[1])/N

peak1_e=np.sum(e1[1])/N
peak2_e=np.sum(e2[1])/N
peak3_e=np.sum(e3[1])/N
peak4_e=np.sum(e4[1])/N
peak5_e=np.sum(e5[1])/N

position_1_a=b1[0][b1[1].argmax()]
position_2_a=b2[0][b2[1].argmax()]
position_3_a=b3[0][b3[1].argmax()]
position_4_a=b4[0][b4[1].argmax()]
position_5_a=b5[0][b5[1].argmax()]

position_1_b=c1[0][c1[1].argmax()]
position_2_b=c2[0][c2[1].argmax()]
position_3_b=c3[0][c3[1].argmax()]
position_4_b=c4[0][c4[1].argmax()]
position_5_b=c5[0][c5[1].argmax()]

position_1_e=e1[0][e1[1].argmax()]
position_2_e=e2[0][e2[1].argmax()]
position_3_e=e3[0][e3[1].argmax()]
position_4_e=e4[0][e4[1].argmax()]
position_5_e=e5[0][e5[1].argmax()]


A1_y=[-2.0,peak1_a]
A1_x=[position_1_a,position_1_a]

A2_y=[-2.0,peak2_a]
A2_x=[position_2_a,position_2_a]

A3_y=[-2.0,peak3_a]
A3_x=[position_3_a,position_3_a]

A4_y=[-2.0,peak4_a]
A4_x=[position_4_a,position_4_a]

A5_y=[-2.0,peak5_a]
A5_x=[position_5_a,position_5_a]


A1_x=np.array(A1_x)
A1_y=np.array(A1_y)
A2_x=np.array(A2_x)
A2_y=np.array(A2_y)

A3_x=np.array(A3_x)
A3_y=np.array(A3_y)
A4_x=np.array(A4_x)
A4_y=np.array(A4_y)

A5_x=np.array(A5_x)
A5_y=np.array(A5_y)


B1_y=[-2.0,peak1_b]
B1_x=[position_1_b,position_1_b]

B2_y=[-2.0,peak2_b]
B2_x=[position_2_b,position_2_b]

B3_y=[-2.0,peak3_b]
B3_x=[position_3_b,position_3_b]

B4_y=[-2.0,peak4_b]
B4_x=[position_4_b,position_4_b]

B5_y=[-2.0,peak5_b]
B5_x=[position_5_b,position_5_b]

B1_x=np.array(B1_x)
B1_y=np.array(B1_y)
B2_x=np.array(B2_x)
B2_y=np.array(B2_y)

B3_x=np.array(B3_x)
B3_y=np.array(B3_y)

B4_x=np.array(B4_x)
B4_y=np.array(B4_y)

B5_x=np.array(B5_x)
B5_y=np.array(B5_y)

E1_y=[-2.0,peak1_e]
E1_x=[position_1_e,position_1_e]
E2_y=[-2.0,peak2_e]
E2_x=[position_2_e,position_2_e]

E3_y=[-2.0,peak3_e]
E3_x=[position_3_e,position_3_e]

E4_y=[-2.0,peak4_e]
E4_x=[position_4_e,position_4_e]

E5_y=[-2.0,peak5_e]
E5_x=[position_5_e,position_5_e]

E1_x=np.array(E1_x)
E1_y=np.array(E1_y)
E2_x=np.array(E2_x)
E2_y=np.array(E2_y)

E3_x=np.array(E3_x)
E3_y=np.array(E3_y)

E4_x=np.array(E4_x)
E4_y=np.array(E4_y)

E5_x=np.array(E5_x)
E5_y=np.array(E5_y)


#%%

fig,ax=plt.subplots()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#ax.set_xscale('log')
#ax.set_yscale('log')
#plt.xlim([-d,d])
width_max=1e1#2e-2/d
#width=5e-1
width_min=5e-4#0.0#-2e-2/d#-2e-3/d
#plt.xlim([width_min,width_max])
c=5.5e0
a=0.5
plt.xlim([-0.3,0.3])
#h=50000
#plt.ylim([-0.2,h])
plt.ylim([0.0,1.1])
my_x_ticks = [-0.2,0.0,0.2]
plt.xticks(my_x_ticks)
my_y_ticks = [0.0,0.5,1.0]
plt.yticks(my_y_ticks)
#my_y_ticks = [0.0,2.0,4.,6.0,8.0,10,12]
#plt.yticks(my_y_ticks)
#my_y_ticks = [0,40,80,120]

#plt.quiver(*origin, V[:,0], V[:,1], color=['r','b'], scale=21)

plt.plot(-2.0,-2.0,'-',color='k'        ,label=r"$U/\Gamma=%g$"%(u1/G),zorder=-10)#
#plt.plot(-2.0,-2.0,'--',color='tab:orange',label=r"$U/\Gamma=%g$"%(u2/G),zorder=-10)#
plt.plot(-2.0,-2.0,':',color='k'  ,label=r"$U/\Gamma=%g$"%(u3/G),zorder=-10)#
#plt.plot(-2.0,-2.0,'--',color='tab:green' ,label=r"$U/\Gamma=%g$"%(u4/G),zorder=-10)#
plt.plot(-2.0,-2.0,'--',color='k'   ,label=r"$U/\Gamma=%g$"%(u5/G),zorder=-10)#

plt.plot(A1_x/d,A1_y*delta_1,'-^',color='tab:red'          ,zorder=-10)#
plt.plot(A3_x/d,A3_y*delta_1,':^',color='tab:red'  ,zorder=-10)#
plt.plot(A5_x/d,A5_y*delta_1,'--^',color='tab:red'   ,zorder=-10)#


plt.plot(B1_x/d,B1_y*delta_2,'-^',color='tab:red'        ,zorder=-10)#
plt.plot(B3_x/d,B3_y*delta_2,':^',color='tab:red'  ,zorder=-10)#
plt.plot(B5_x/d,B5_y*delta_2,'--^',color='tab:red'   ,zorder=-10)#

plt.plot(E1_x/d,E1_y*delta_3,'-^',color='tab:blue'        ,zorder=-10)#
plt.plot(E3_x/d,E3_y*delta_3,':^',color='tab:blue'  ,zorder=-10)#
plt.plot(E5_x/d,E5_y*delta_3,'--^',color='tab:blue'   ,zorder=-10)#

plt.plot(-E1_x/d,E1_y*delta_3,'-^',color='tab:cyan'        ,zorder=-10)#
plt.plot(-E3_x/d,E3_y*delta_3,':^',color='tab:cyan'  ,zorder=-10)#
plt.plot(-E5_x/d,E5_y*delta_3,'--^',color='tab:cyan'   ,zorder=-10)#


plt.plot(-A1_x/d,A1_y*delta_1,'-^',color='tab:green'          ,zorder=-10)#
plt.plot(-A3_x/d,A3_y*delta_1,':^',color='tab:green'  ,zorder=-10)#
plt.plot(-A5_x/d,A5_y*delta_1,'--^',color='tab:green'   ,zorder=-10)#


plt.plot(-B1_x/d,B1_y*delta_2,'-^',color='tab:green'        ,zorder=-10)#
plt.plot(-B3_x/d,B3_y*delta_2,':^',color='tab:green'  ,zorder=-10)#
plt.plot(-B5_x/d,B5_y*delta_2,'--^',color='tab:green'   ,zorder=-10)#

#ax.text(position_3_a/d, peak3_a*d,'^',fontsize=13.5,zorder=5)


plt.legend(loc='best',fontsize=12.5)
plt.xlabel(r'$\omega/\Delta$',fontsize=18)
plt.ylabel(r'$\Delta\rho(\omega)$',fontsize=18)
#plt.title(r"$t$=%g,$\Delta$=%g,$\epsilon$=%g,$W$=%g"%(T,d,E,W),fontsize=14)



plt.tight_layout()
plt.savefig('LDOS_FRG.png',format='png',dpi=300)

#plt.savefig('plot/plot.png',format='png',dpi=300)
#plt.savefig('plot/SSH_TN_metallic_RGflow_finite_bias_a.pdf',format='pdf',dpi=300)

#plt.tight_layout()
#plt.savefig('plot/IRLM_HF_RGflow_twoSSH_TN_changing_t_k.pdf',format='pdf',dpi=300)
plt.show()   




