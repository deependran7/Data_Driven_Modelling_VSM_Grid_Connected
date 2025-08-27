"""
Non_Linear_Model.py 
Solution of non linear equation of system of equation of inverter.
Dependencies:
    numpy, sympy,scipy

Usage:
    # Given X0, initial state
    # returns the derivative of states for the solution of non linear equations of VSM based inverter
"""

from numpy import sqrt, sin, cos, pi
import numpy as np

Rf = 0.07;
Lf = 5.2e-3;
Cf = 100e-6;
Lc = 1.5e-4;
Rc = 0.05;

Kpv,Kiv,Kpc,Kic = 3,400,5,10;

J=0.05;
D=10;
wn=2*pi*50;
wf = 10*pi;
mq = 1e-5;
Kppll = 180;
Kipll = 3200;

def Inverter_Model(t,x,Pref):
    vod,voq,ild,ilq,iod,ioq,Pinv,Qinv,dwVSM,aVSM,deltaVSM,gammad,gammaq,zetad,zetaq = x
    #grid model   
    vbD,vbQ = 400*sqrt(2/3),0
    #ref setpoint
    Vdref,Vqref = 400*sqrt(2/3),0
    wn = 2*pi*50
    #Pref = 50000
    Qref = 0
    #Power calculation
    pinv = (3/2)*(vod*iod + voq*ioq)
    qinv = (3/2)*(voq*iod - vod*ioq)
    Pinv_dot = - wf*Pinv + wf*pinv 
    Qinv_dot = - wf*Qinv + wf*qinv 
    #voltage control loop
    vodrefin = Vdref + (Qref-Qinv)*mq
    voqrefin = Vqref
    #frequency control loop
    dwVSM_dot = (((Pref-Pinv)/wn)-D*(dwVSM))/J
    wVSM = wn+dwVSM
    aVSM_dot = wVSM
    deltaVSM_dot = wVSM - wn #dwVSM
    #reftransfer
    vbd,vbq = cos(deltaVSM)*vbD + sin(deltaVSM)*vbQ, -sin(deltaVSM)*vbD + cos(deltaVSM)*vbQ
    #Virtual impedance
    vodref = vodrefin - ild*Rf + ilq*wVSM*Lf
    voqref = voqrefin - ilq*Rf - ild*wVSM*Lf
    #voltage Controller
    gammad_dot = vodref - vod
    gammaq_dot = voqref - voq
    ifdref  = Kpv*(vodref - vod) + Kiv*gammad + iod - wVSM*Cf*voq
    ifqref  = Kpv*(voqref - voq) + Kiv*gammaq + ioq + wVSM*Cf*vod
    #current Controller
    zetad_dot = ifdref - ild
    zetaq_dot = ifqref - ilq
    vfdref = Kpc*(ifdref - ild) + Kic*zetad + vod - wVSM*Lf*ilq
    vfqref = Kpc*(ifqref - ilq) + Kic*zetaq + voq + wVSM*Lf*ild
    #Inverter Model
    vid,viq = np.array([vfdref,vfqref])
    #Filter Model
    ild_dot = (-Rf/Lf)*ild + wVSM*ilq + (vid - vod)/Lf
    ilq_dot = -wVSM*ild + (-Rf/Lf)*ilq + (viq - voq)/Lf
    vod_dot = wVSM*voq + (ild - iod)/Cf
    voq_dot = -wVSM*vod + (ilq - ioq)/Cf
    #Coupling Line
    iod_dot = (-Rc/Lc)*iod + wVSM*ioq + (vod - vbd)/Lc
    ioq_dot = -wVSM*iod + (-Rc/Lc)*ioq + (voq - vbq)/Lc
    
    return np.array([vod_dot,voq_dot,ild_dot,ilq_dot,iod_dot,ioq_dot,Pinv_dot,Qinv_dot,dwVSM_dot,aVSM_dot,deltaVSM_dot,gammad_dot,gammaq_dot,zetad_dot,zetaq_dot])