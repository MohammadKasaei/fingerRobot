from cProfile import label
import pstats
import numpy as np
import control
from   scipy.integrate import solve_ivp
import scipy.sparse as sparse

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

import osqp

class SFODE():
    def __init__(self) -> None:
        
        self.ts = 0.041
        self.Np = 10
        self.defineSystem()



    def defineSystem(self):
        self.A = np.matrix([[0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0]])

        self.B = np.matrix ([[0,0,0],
                             [1,0,0],
                             [0,0,0],
                             [0,1,0],
                             [0,0,0],
                             [0,0,1]])
        self.C = np.eye(self.A.shape[0])
        self.D = 0*self.B
        
        sys = control.ss(self.A,self.B,self.C,self.D)
        sys_d = control.c2d (sys,self.ts)

        self.Ad = np.matrix(sys_d.A)
        self.Bd = np.matrix(sys_d.B)

    def uPD(self):
        k = np.matrix([[100, 50, 0, 0, 0 , 0],
                [0, 0, 100, 50, 0 , 0],
                [0, 0, 0, 0, 100 , 50]]) 
        u = -k@(self.states-self.ref) 
        return u

        
    def u_MPC(self):

        [nx, nu] = self.Bd.shape # number of states and number or inputs

        # Constraints
        uref = 0
        uinit = 0 # not used here
        # umin = np.array([-5.0,-5.0,-5.0]) 
        # umax = np.array([ 5.0, 5.0, 5.0]) 

        # xmin = np.array([-2.0, -10, -2.0,-10.0,-2.0,-10.0])
        # xmax = np.array([ 2.0,  10,  2.0, 10.0, 2.0, 10.0])
        
        umin = 10*np.array([-0.01,-0.01,-0.01]) 
        umax = 10*np.array([ 0.01, 0.01, 0.01]) 

        # xmin = np.array([-0.05, -0.1, -0.05,-0.1,-0.05,-0.1])
        # xmax = -np.copy(xmin) #np.array([-0.05, -0.01, -0.05,-0.01,-0.05,-0.01])
        xmin = np.array([-0.05, -0.1, -0.05, -0.1,  0.0 , -0.1])
        xmax = np.array([ 0.05 , 0.1,  0.05,  0.1,  0.15,  0.1])
        
        # Objective function
        Q = sparse.diags([50.0, 1.0, 50.0, 1.0,50.0, 1.0])
        QN = sparse.diags([100.0, 0.0, 100.0, 0.0,100.0, 0.0]) # final cost
        R = 0.001*sparse.eye(nu)

        # Initial and reference states
        x0 = np.array(self.states)#np.array(x_0) #np.array([0.1, 0.2]) # initial state

        # Reference input 
        xref = self.ref 

        # Prediction horizon
        Np = self.Np

        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # - quadratic objective
        P = sparse.block_diag([sparse.kron(sparse.eye(Np), Q), QN,
                            sparse.kron(sparse.eye(Np), R)]).tocsc()
        # - linear objective
        # q = np.hstack([np.kron(np.ones(Np), -Q.dot(xref)), -QN.dot(xref),
        #             np.zeros(Np * nu)])
        
        refTrajCost = []
        for i in range(Np):
            refTrajCost = np.append(refTrajCost,-Q.dot(xref[:,i]),axis =0)

        q = np.hstack([refTrajCost, -QN.dot(xref[:,Np-1]),
                    np.zeros(Np * nu)])

        # - linear dynamics
        Ax = sparse.kron(sparse.eye(Np + 1), -sparse.eye(nx)) + sparse.kron(sparse.eye(Np + 1, k=-1), self.Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, Np)), sparse.eye(Np)]), self.Bd)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-x0, np.zeros(Np * nx)])
        ueq = leq # for equality constraints -> upper bound  = lower bound!

        # - input and state constraints
        Aineq = sparse.eye((Np + 1) * nx + Np * nu)
        lineq = np.hstack([np.kron(np.ones(Np + 1), xmin), np.kron(np.ones(Np), umin)]) # lower bound of inequalities
        uineq = np.hstack([np.kron(np.ones(Np + 1), xmax), np.kron(np.ones(Np), umax)]) # upper bound of inequalities
        
        # - OSQP constraints
        A = sparse.vstack([Aeq, Aineq]).tocsc()
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])

        # Create an OSQP object
        prob = osqp.OSQP()

        # Setup workspace
        prob.setup(P, q, A, l, u, warm_start=True)

        res = prob.solve()
        # Apply first control input to the plant
        uMPC = res.x[-Np * nu:-(Np - 1) * nu]
        return uMPC



    def ode_fcn(self,t,x):
        # u    = self.u
        # k = np.matrix([[100, 50, 0, 0, 0 , 0],
        #                [0, 0, 100, 50, 0 , 0],
        #                [0, 0, 0, 0, 100 , 50]]) 

        # u = -k@(self.states-self.ref) 
        # u = self.uPD()        
        u = np.matrix(self.u_MPC())
        
        checkNone = True if u[0,0] is None or u[0,1] is None or u[0,2] is None else False
        if checkNone:
            u = self.lastU            
        self.lastU = np.copy(u)
        
        # u = np.clip(u, -5,5)
        self.u = np.copy(u)


        dxdt = self.A@(np.matrix(x).T) + self.B@(u.T)
        # dxdt = self.Ad@(np.matrix(x).T) + self.Bd@(u.T)
        
        return dxdt.T


    def ode_step(self):
        stime          = (self.sim_time,self.sim_time+self.ts)
        t_eval         = np.array([stime[1]]) #np.linspace(t0, tfinal, int(tfinal/ts))
        sol            = solve_ivp(self.ode_fcn,stime,self.states,t_eval=t_eval)
        self.sim_time += self.ts
        self.states    = [y[0] for y in sol.y]
        #print ("t: {0}, states: {1}".format(self.sim_time,self.states))
        return self.states


    def run_sim_steps(self,ref,x0,t0,ts,tfinal):  
        
        y = x0
        self.states = np.copy(x0)
        for n in range(int(tfinal/ts)):
            tfinal   = t0+ts
            sim_time = (t0,tfinal)
            self.ts  = ts
            t_eval   = np.array([t0+ts])#np.linspace(t0, tfinal, int(tfinal/ts))
            # self.ref = np.array([np.sin(t0),np.cos(t0),np.cos(t0),-np.sin(t0), t0,1]) #ref
            self.ref = np.array([1,0.0,2,0., 1.5,0.]) #ref
            
            sol      = solve_ivp(self.ode_fcn,sim_time,x0,t_eval=t_eval)
            x0 = [y[0] for y in sol.y]
            self.states    = [y[0] for y in sol.y]
           

            t0 +=ts
        
            if (n):
                yn = np.array([y[0] for y in sol.y])
                y = np.append(y,[yn],axis=0)
            else:
                y = [np.array([y[0] for y in sol.y])]
                
        self.animation_states = y.T

        return y.T

    def singleODEStep(self,):
        stime          = (self.sim_time,self.sim_time+self.ts)
        t_eval         = np.array([stime[1]]) #np.linspace(t0, tfinal, int(tfinal/ts))
        sol            = solve_ivp(self.ode_fcn,stime,self.states,t_eval=t_eval)
        self.sim_time += self.ts
        self.states    = [y[0] for y in sol.y]
        
        return np.array(self.states)

    def calcLengthBackbone(self,x0,t0,ts,tfinal,ref):
        
        y   = x0
        stc = 0
        self.states = np.copy(x0)
        i=0

        for n in range(int(tfinal/ts)):
            tfinal   = t0+ts
            sim_time = (t0,tfinal)
            self.ts  = ts
            t_eval   = np.array([t0+ts])#np.linspace(t0, tfinal, int(tfinal/ts))
            # self.ref = np.array([np.sin(t0),np.cos(t0),np.cos(t0),-np.sin(t0), t0,1]) #ref

            self.ref = ref[:,i:i+self.Np+1]
            i+=1
            
            sol      = solve_ivp(self.ode_fcn,sim_time,x0,t_eval=t_eval)
            x0 = [y[0] for y in sol.y]
            self.states    = [y[0] for y in sol.y]

            stc += np.sqrt(self.states[1]**2 + self.states[3]**2 + self.states[5]**2)*sys.ts
            

            t0 +=ts
            if (n):
                yn = np.array([y[0] for y in sol.y])
                y = np.append(y,[yn],axis=0)
            else:
                y = [np.array([y[0] for y in sol.y])]

        return stc,y.T
        





if __name__ == "__main__":
    
    
    sys = SFODE()
    
    x0 =  np.array([0,0,0,0,0.1,0])    
    sys.states = np.copy(x0)
    y = [np.array(x0)]    
    u = [np.array([0,0,0])]

    tf = 1
    tfinal = tf+(sys.Np*sys.ts)
    amp = [0.02,0.02,0.00]
    sys.sim_time = 1
    tt = np.linspace(0,tfinal,int(tfinal/sys.ts))
    sys.ref = np.vstack ([amp[0]*np.sin(tt),amp[0]*np.cos(tt),amp[1]*(1-np.cos(tt)),amp[1]*np.sin(tt), x0[4] + amp[2]*tt, amp[2]*np.ones(int(tfinal/sys.ts))])
    
    t0=0
    
    # stc, y = sys.calcLengthBackbone(x0,t0,sys.ts,tfinal-(sys.Np*sys.ts),sys.ref) #:(tt,sys.ref,x0)
    stc, y = sys.calcLengthBackbone(x0,t0,sys.ts,tf,sys.ref) #:(tt,sys.ref,x0)

    tt = np.linspace(0,tfinal-(sys.Np*sys.ts),int((tfinal-(sys.Np*sys.ts))/sys.ts))

    plt.plot(tt,y[0,:],'r',label='x')
    plt.plot(tt,amp[0]*np.sin(tt),'r--',label='xr')
    plt.plot(tt,y[2,:],'g',label='y')
    plt.plot(tt,amp[1]*(1-np.cos(tt)),'g--',label='yr')
    plt.plot(tt,y[4,:],'b',label='z')
    plt.plot(tt,x0[4] + amp[2]*tt,'b--',label='z')
    # plt.plot(tt,np.sqrt(0.02*2)*tt)
    plt.grid()
    plt.legend()
    plt.show()



    sys.sim_time = 0
    tfinal = 5

    st = np.array([0])
    rrr = [np.array([0,0,0,0])]
    stp = 0
    stp_xz = 0
    stp_yz = 0

    amp = [0.02,0.02,0.01]

    for _ in range (int(tfinal/sys.ts)):   
        t = np.linspace(sys.sim_time,np.clip(sys.sim_time+(sys.Np*sys.ts),0,tfinal),sys.Np)
        sys.ref = np.vstack ([amp[0]*np.sin(t),amp[0]*np.cos(t),amp[1]*(1-np.cos(t)),amp[1]*np.sin(t), x0[4] + amp[2]*t, amp[2]*np.ones(sys.Np)])

        # sys.ref = amp*np.array([np.sin(sys.sim_time),np.cos(sys.sim_time),1-np.cos(sys.sim_time),np.sin(sys.sim_time), sys.sim_time,1])             
        # sys.ref = amp*np.array([np.sin(sys.sim_time),np.cos(sys.sim_time),1-np.cos(sys.sim_time),np.sin(sys.sim_time), sys.sim_time,1])     
        yn = sys.singleODEStep()
        y = np.append(y,[yn],axis=0)
        u = np.append(u,sys.u,axis=0)


        stc = np.sqrt(yn[1]**2 + yn[3]**2 + yn[5]**2)*sys.ts
        stc_xz = np.sqrt(yn[1]**2 + yn[5]**2)*sys.ts      
        stc_yz = np.sqrt(yn[3]**2 + yn[5]**2)*sys.ts
        
        sBackbone = stc+stp
        
        sBackbone_xz = stc_xz + stp_xz
        sBackbone_yz = stc_yz + stp_yz
        
        st = np.append (st,[sBackbone],axis=0)
        stp = sBackbone
        
        stp_xz = sBackbone_xz
        stp_yz = sBackbone_yz
        
        delta = 0.0075

        # r1 = sBackbone_xz+np.array([yn[1],yn[3],yn[5]]).dot(np.array([delta*np.cos(0*np.pi/2), delta*np.sin(0*np.pi/2),0 ]))
        # r2 = sBackbone_yz+np.array([yn[1],yn[3],yn[5]]).dot(np.array([delta*np.cos(1*np.pi/2), delta*np.sin(1*np.pi/2),0 ]))
        # r3 = sBackbone_xz+np.array([yn[1],yn[3],yn[5]]).dot(np.array([delta*np.cos(2*np.pi/2), delta*np.sin(2*np.pi/2),0 ]))
        # r4 = sBackbone_yz+np.array([yn[1],yn[3],yn[5]]).dot(np.array([delta*np.cos(3*np.pi/2), delta*np.sin(3*np.pi/2),0 ]))
        r1 = sBackbone_xz+np.array([yn[1],yn[3],yn[5]]).dot(np.array([delta*np.cos(0*np.pi/2), delta*np.sin(0*np.pi/2),0 ]))
        r2 = sBackbone_yz+np.array([yn[1],yn[3],yn[5]]).dot(np.array([delta*np.cos(1*np.pi/2), delta*np.sin(1*np.pi/2),0 ]))
        r3 = sBackbone_xz+np.array([yn[1],yn[3],yn[5]]).dot(np.array([delta*np.cos(2*np.pi/2), delta*np.sin(2*np.pi/2),0 ]))
        r4 = sBackbone_yz+np.array([yn[1],yn[3],yn[5]]).dot(np.array([delta*np.cos(3*np.pi/2), delta*np.sin(3*np.pi/2),0 ]))
        
        # robot.sendMessage ([int(r1*1000),int(r2*1000),int(r3*1000),int(r4*1000)]) 

        rrr = np.append (rrr,[np.array([r1,r2,r3,r4])],axis=0)

        # stc = np.sqrt(yn[1]**2 + yn[3]**2 + yn[5]**2)*sys.ts
        
        # sBackbone = stc+stp
        # st = np.append (st,[sBackbone],axis=0)
        # stp = sBackbone
        
        # delta = 0.05

        # r1 = sBackbone+np.array([yn[1],yn[3],yn[5]]).dot(np.array([delta*np.cos(0*np.pi/2), delta*np.sin(0*np.pi/2),0 ]))
        # r2 = sBackbone+np.array([yn[1],yn[3],yn[5]]).dot(np.array([delta*np.cos(1*np.pi/2), delta*np.sin(1*np.pi/2),0 ]))
        # r3 = sBackbone+np.array([yn[1],yn[3],yn[5]]).dot(np.array([delta*np.cos(2*np.pi/2), delta*np.sin(2*np.pi/2),0 ]))
        # r4 = sBackbone+np.array([yn[1],yn[3],yn[5]]).dot(np.array([delta*np.cos(3*np.pi/2), delta*np.sin(3*np.pi/2),0 ]))

        # rrr = np.append (rrr,[np.array([r1,r2,r3,r4])],axis=0)



    fig, axs = plt.subplots(4)
    
    t = np.linspace(0,tfinal,y.shape[0])
    
    axs[0].plot(t,y[:,0],'ro',label='x')    
    axs[0].plot(t,y[:,1],'r',label='xdot')
    axs[0].plot(t,amp[0]*np.sin(t),'k--',lw=3,label='xref')
    axs[0].plot(t,u[:,0],'m--',lw=3,label='ux')
    
    

    axs[0].legend()
    axs[0].grid()
    
    
    axs[1].plot(t,y[:,2],'go',label='y')
    axs[1].plot(t,y[:,3],'g--',label='ydot')
    axs[1].plot(t,amp[1]*(1-np.cos(t)),'k--',lw=3,label='yref')
    axs[1].plot(t,u[:,1],'m--',lw=3,label='uy')


    axs[1].legend()
    axs[1].grid()
    
    axs[2].plot(t,y[:,4],'bo',label='z')
    axs[2].plot(t,y[:,5],'b--',label='zdot')
    axs[2].plot(t,x0[4]+amp[2]*t,'k--',lw=3,label='zref')
    axs[2].plot(t,u[:,2],'m--',lw=3,label='uz')

    axs[2].legend()
    axs[2].grid()

    axs[3].plot(t,st,'bo',label='st')
    axs[3].plot(t,rrr[:,0],'r',label='r1')
    axs[3].plot(t,rrr[:,1],'g',label='r2')
    axs[3].plot(t,rrr[:,2],'b',label='r3')
    axs[3].plot(t,rrr[:,3],'k',label='r4')
        
   
    axs[3].legend()
    axs[3].grid()

    plt.show()
       

    fig = plt.figure()
    plt.plot(t,(1000*rrr[:,0]).astype(int),'r',label='r1')
    plt.plot(t,(1000*rrr[:,1]).astype(int),'g',label='r2')
    plt.plot(t,(1000*rrr[:,2]).astype(int),'b',label='r3')
    plt.plot(t,(1000*rrr[:,3]).astype(int),'k',label='r4')
    plt.legend()
    plt.grid()
    plt.show()

    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter (y[:,0],y[:,2],y[:,4],'r',label='robot')
    ax.scatter (amp[0]*np.sin(t),amp[1]*(1-np.cos(t)),x0[4]+(amp[2]*t),'k',label='ref')
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    ax.set_zlim(-0.0,0.15)
    ax.legend()
    
    plt.show()

    

  