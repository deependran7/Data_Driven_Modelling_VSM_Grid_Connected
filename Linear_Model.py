"""
Linear_Model.py 
Linearization of the nonlinear VSM inverter model around an operating point.
Dependencies:
    numpy, sympy,scipy

Usage:
    # Given X0, initial state
    # returns A,B matrices of linearized system
"""

from sympy import symbols,sin,cos,Matrix,diff,zeros
import numpy as np


Rfi = 0.07;
Lfi = 5.2e-3;
Cfi = 100e-6;
Lci = 1.5e-4;
Rci = 0.05;

Kpvi,Kivi,Kpci,Kici = 3,400,5,10;

Ji=0.05;
Di=10;
wni=2*np.pi*50;
wfi = 10*np.pi;
mqi = 1e-5;
vbDi,vbQi = 400*np.sqrt(2/3),0
#ref setpoint
Vdrefi,Vqrefi = 400*np.sqrt(2/3),0
wni = 2*np.pi*50
Prefi = 50000
Qrefi = 0

#dwVSMsi,aVSMsi,deltaVSMsi,pinvsi,qinvsi,gammadsi,gammaqsi,zetadsi,zetaqsi,ildsi,ilqsi,vodsi,voqsi,iodsi,ioqsi= np.array([ 8.69760127e+00,  1.63100410e+01,  6.02077755e-01,  2.66882107e+04,
#       -3.89359734e+03,  7.51036228e-03, -3.94805175e-03,  1.46976875e-01,
#       -3.20960660e-02,  1.10541602e+02, -2.67057977e+01,  2.75977488e+02,
#       -1.81903169e+02,  1.04516795e+02, -3.52424182e+01])



dwVSM,aVSM,deltaVSM,Pinv,Qinv,gammad,gammaq,zetad,zetaq,ild,ilq,vod,voq,iod,ioq = symbols(['\Delta_omega_VSM','aVSM','delta_VSM','P_inv','Q_inv','gamma_d','gamma_q','zeta_d','zeta_q','i_ld','i_lq','v_od','v_oq','i_od','i_oq'])
vbD,vbQ=symbols(['v_bD','v_bQ'])
Rf,Cf,Lf,Rc,Lc = symbols(['R_f','C_f','L_f','R_c','L_c'])
Kpc,Kic,Kpv,Kiv = symbols(['K_pc','K_ic','K_pv','K_iv'])
Vdref,Vqref,wn,wf,wr,wVSM = symbols(['V_dref','V_qref','omega_n','omega_f','omega_r','omega_vsm'])
Pref,Qref = symbols(['P_ref','Q_ref'])
D,J,mq,ka = symbols(['D','J','m_q','k_alpha'])


def get_matrix(der,stat):
    mat = zeros(len(der),len(stat))
    for i in range(len(der)):
        for j in range(len(stat)):
            mat[i,j] = diff(der[i],stat[j])
    return mat

def linear_model(X0):
    #Power calculation
    pinv = (3/4)*(vod*iod + voq*ioq)
    qinv = (3/4)*(voq*iod - vod*ioq)
    Pinv_dot = - wf*Pinv + wf*pinv 
    Qinv_dot = - wf*Qinv + wf*qinv 
    #voltage control loop
    vodrefin = Vdref + (Qref-Qinv)*mq
    voqrefin = Vqref
    #frequency control loop
    dwVSM_dot = (((Pref-Pinv)/wn)-D*(dwVSM))/J
    wVSM = wr+dwVSM
    aVSM_dot = wVSM
    deltaVSM_dot = dwVSM
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

    Xinv = Matrix([vod,voq,ild,ilq,iod,ioq,Pinv,Qinv,dwVSM,aVSM,deltaVSM,gammad,gammaq,zetad,zetaq])
    Xinv_dot = Matrix([vod_dot,voq_dot,ild_dot,ilq_dot,iod_dot,ioq_dot,Pinv_dot,Qinv_dot,dwVSM_dot,aVSM_dot,deltaVSM_dot,gammad_dot,gammaq_dot,zetad_dot,zetaq_dot])
    Uinv = Matrix([Pref,Qref,Vdref,Vqref,wr,vbD,vbQ])
    Vinv = Matrix([vbD,vbQ])

    Ainv,Binv,Cinv = get_matrix(Xinv_dot,Xinv),get_matrix(Xinv_dot,Uinv),get_matrix(Xinv_dot,Vinv)

    vodsi,voqsi,ildsi,ilqsi,iodsi,ioqsi,pinvsi,qinvsi,dwVSMsi,aVSMsi,deltaVSMsi,gammadsi,gammaqsi,zetadsi,zetaqsi = X0

    Amm = Ainv.subs([(Kpv,Kpvi),(Kiv,Kivi),(Kpc,Kpci),(Kic,Kici),(Lf,Lfi),(Rf,Rfi),(Cf,Cfi),(Rc,Rci),(Lc,Lci),(wn,wni),(wr,wni)])
    Amm = Amm.subs([(ild,ildsi),(ilq,ilqsi),(deltaVSM,deltaVSMsi),(vod,vodsi),(voq,voqsi),(iod,iodsi),(ioq,ioqsi)])
    Amm = Amm.subs([(dwVSM,dwVSMsi),(mq,mqi),(D,Di),(J,Ji),(wf,wfi),(vbD,vbDi),(vbQ,vbQi)])
    Bmm = Binv.subs([(Kpv,Kpvi),(Kiv,Kivi),(Kpc,Kpci),(Kic,Kici),(Lf,Lfi),(J,Ji),(wn,wni),(mq,mqi),(Lc,Lci),(deltaVSM,deltaVSMsi)]) 
    Bmm = Bmm.subs([(Cf,Cfi), (Pinv,pinvsi), (Pref,Prefi), (ild,ildsi), (ilq,ilqsi), (iod,iodsi), (ioq,ioqsi), (vod,vodsi), (voq,voqsi)])
    Cmm = Cinv.subs([(deltaVSM,deltaVSMsi),(Lc,Lci)])
    Am = np.array(Amm).astype(np.float64)
    Bm = np.array(Bmm).astype(np.float64)
    Cm = np.array(Cmm).astype(np.float64)
    
    #ut = np.array([vbDi,vbQi])
    return Am,Bm


