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
        
        self.ts = 0.02
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
        umin = np.array([-0.01,-0.01,-0.01]) 
        umax = np.array([ 0.01, 0.01, 0.01]) 

        xmin = np.array([-0.05, -0.01, -0.05,-0.01,-0.05,-0.01])
        # xmax = np.array([-0.05, -0.01, -0.05,-0.01,-0.05,-0.01])
        xmax = - xmin #np.array([-0.05, -0.01, -0.05,-0.01,-0.05,-0.01])
        

        # Objective function
        Q = sparse.diags([50.0, 10.0, 50.0, 10.0,50.0, 10.0])
        
        QN = sparse.diags([100.0, 0.0, 100.0, 0.0,100.0, 0.0]) # final cost
        R = 0.001*sparse.eye(nu)

        # Initial and reference states
        x0 = np.array(self.states)#np.array(x_0) #np.array([0.1, 0.2]) # initial state

        # Reference input 
        xref = self.ref 

        # Prediction horizon
        Np = 10

        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # - quadratic objective
        P = sparse.block_diag([sparse.kron(sparse.eye(Np), Q), QN,
                            sparse.kron(sparse.eye(Np), R)]).tocsc()
        # - linear objective
        q = np.hstack([np.kron(np.ones(Np), -Q.dot(xref)), -QN.dot(xref),
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
        
        
        # u = np.clip(u, -0.05,5)
        
        self.u = np.copy(u)

        dxdt = self.A@(np.matrix(x).T) + self.B@(u.T)
        
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


  

if __name__ == "__main__":
    sys = SFODE()
    # ref = np.array([0,0,0])
    # x0 =  np.array([0,0,0,0,0,0])
    # t0 = 0
    ts = sys.ts
    tfinal = 3

    # y = sys.run_sim_steps(ref,x0,t0,ts,tfinal)
    
    
    # t = np.linspace(0,tfinal,y.shape[1])
    # plt.plot(t,y[0,:],'ro',label='x')
    # plt.plot(t,y[1,:],'r--',label='xdot')
    
    # plt.plot(t,y[2,:],'go',label='y')
    # plt.plot(t,y[3,:],'g--',label='ydot')
    
    # plt.plot(t,y[4,:],'bo',label='z')
    # plt.plot(t,y[5,:],'b--',label='zdot')
    
    # plt.legend()
    # plt.xlabel("Time(s)")
    # plt.ylabel("Pos(m)")    
    # plt.show()
    

    x0 =  np.array([0,0,0,0,0,0])    
    sys.states = np.copy(x0)
    y = [np.array(x0)]    
    sys.ref = np.array([1,0.0,2,0., 1.5,0.])
    sys.sim_time = 0
    tfinal = 3

    for _ in range (int(tfinal/sys.ts)):   
        sys.ref = 0.05*np.array([np.sin(sys.sim_time),np.cos(sys.sim_time),1-np.cos(sys.sim_time),np.sin(sys.sim_time), sys.sim_time,1])     
        yn = sys.singleODEStep()
        y = np.append(y,[yn],axis=0)
        
        

    fig, axs = plt.subplots(3)
    
    t = np.linspace(0,tfinal,y.shape[0])
    
    axs[0].plot(t,y[:,0],'ro',label='x')    
    axs[0].plot(t,y[:,1],'r',label='xdot')
    axs[0].plot(t,0.05*np.sin(t),'k--',lw=3,label='xref')
    

    axs[0].legend()
    axs[0].grid()
    
    
    axs[1].plot(t,y[:,2],'go',label='y')
    axs[1].plot(t,y[:,3],'g--',label='ydot')
    axs[1].plot(t,0.05*(1-np.cos(t)),'k--',lw=3,label='yref')

    axs[1].legend()
    axs[1].grid()
    
    axs[2].plot(t,y[:,4],'bo',label='z')
    axs[2].plot(t,y[:,5],'b--',label='zdot')
    axs[2].plot(t,0.05*t,'k--',lw=3,label='zref')

    axs[2].legend()
    axs[2].grid()
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter (y[:,0],y[:,2],y[:,4],'r',label='robot')
    ax.scatter (0.05*np.sin(t),0.05*(1-np.cos(t)),0.05*t,'k',label='ref')
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.legend()
    
    plt.show()

    

