import matplotlib
import numpy as np
# from ODEGym import FingerRobotOdeEnv
from   scipy.integrate import solve_ivp
import scipy.sparse as sparse

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


class SoftRobotControl():
    def __init__(self) -> None:
        # initial length of robot
        self.l0 = 0.101
        # cables offset
        self.d  = 7.5e-3
        # ode step time
        self.ds     = 0.0005  

        r0 = np.array([0,0,0]).reshape(3,1)  
        R0 = np.eye(3,3)
        R0 = np.reshape(R0,(9,1))
        y0 = np.concatenate((r0, R0), axis=0)
        self.states = np.squeeze(np.asarray(y0))
        self.y0 = np.copy(self.states)

    def Jac(self, f, q, dq=np.array((1e-3,1e-3,1e-3))):
        fx0 = f(q)
        n   = len(q)
        m   = len(fx0)
        jac = np.zeros((n, m))
        for j in range(m):  # through rows 
            if (j==0):
                Dq = np.array((dq[0]/2.0,0,0))
            elif (j==1):
                Dq = np.array((0,dq[1]/2.0,0))
            elif (j==2):
                Dq = np.array((0,0,dq[2]/2.0))

            jac [j,:] = (f(q+Dq) - f(q-Dq))/dq[j]
        return jac    

    def f(x):
        return np.array([np.sin(x[0]),np.cos(x[1]),np.sin(x[1]+np.cos(x[2]))])


    def odeFunction(self,s,y):
            dydt  = np.zeros(12)
            # % 12 elements are r (3) and R (9), respectively
            e3    = np.array([0,0,1]).reshape(3,1)              
            u_hat = np.array([[0,0,self.uy], [0, 0, -self.ux],[-self.uy, self.ux, 0]])
            r     = y[0:3].reshape(3,1)
            R     = np.array( [y[3:6],y[6:9],y[9:12]]).reshape(3,3)
            # % odes
            dR  = R @ u_hat
            dr  = R @ e3
            dRR = dR.T
            dydt[0:3]  = dr.T
            dydt[3:6]  = dRR[:,0]
            dydt[6:9]  = dRR[:,1]
            dydt[9:12] = dRR[:,2]
            return dydt.T

    def runOdeForJac(self,q):        
        l  = self.l0 + q[0]
        self.uy = (q[1]) / (l *self.d)
        self.ux = (q[2]) / -(l*self.d)

        cableLength          = (0,l)
        t_eval               = np.linspace(0, l, int(l/self.ds))
        sol                  = solve_ivp(self.odeFunction,cableLength,self.y0,t_eval=t_eval)
        self.states          = np.squeeze(np.asarray(sol.y[:,-1]))
        #print ("t: {0}, states: {1}".format(self.sim_time,self.states))
        return self.states[0:3]

    def visualize(self,state):
        fig = plt.figure()
        ax  = fig.add_subplot(projection='3d')
        ax.scatter (state[:,0],state[:,1],state[:,2], c = 'g', label='robot')
        ax.plot3D(state[:,0], state[:,1],state[:,2], c='r',lw=2)

        
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')
        # ax.set_xlim(-0.08, 0.08)
        # ax.set_ylim(-0.06, 0.06)
        # ax.set_zlim(-0.05, 0.15)
        
        ax.legend()
        
        plt.show()


if __name__ == "__main__":
    env = SoftRobotControl()
    # q = np.array([0.0,-0.0,-0.0])
    # x = env.runOdeForJac(q)
    # print (x)
   
    q = np.array([0.0,-0.0,-0.0])

    xdot = np.array((0.05,0.05,0.0))

    ts = 0.01
    logState = np.array([])
    for i in range(100):    
        jac = env.Jac(env.runOdeForJac,q).T
        pseudo_inverse = np.linalg.pinv(jac)
        qdot = pseudo_inverse @ xdot        
        q   += (qdot * ts)
        
        # plt.plot(i*ts,q[0],'r*')
        # plt.plot(i*ts,q[1],'go')
        # plt.plot(i*ts,q[2],'bx')

        plt.plot(i*ts,q[0]+q[1],'r*')
        plt.plot(i*ts,q[0]-q[1],'g*')
        plt.plot(i*ts,q[0]+q[2],'b*')
        plt.plot(i*ts,q[0]-q[2],'k*')
        
        # plt.plot(i*ts,env.states[0],'r*')
        # plt.plot(i*ts,env.states[1],'g*')
        # plt.plot(i*ts,env.states[2],'b*')
        
        if i==0:
            logState = np.copy(env.states)
            plt.grid()
        else:
            logState =  np.vstack((logState,env.states))
        

        plt.pause(ts)

        
        
    plt.show()    

        

    env.visualize(logState)
        








