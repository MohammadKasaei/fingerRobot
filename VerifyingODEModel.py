import numpy as np
from   scipy.integrate import solve_ivp
import scipy.sparse as sparse

import matplotlib.pyplot as plt

import gym
from   gym import spaces
from   numpy.core.function_base import linspace

from   stable_baselines3.common.env_util import make_vec_env
from   stable_baselines3 import PPO
from   stable_baselines3.common.utils import set_random_seed

from   stable_baselines3.common.vec_env import SubprocVecEnv
from   stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from mpl_toolkits.mplot3d import Axes3D 



import time
from   torch import dsmm

class FingerRobotOdeEnv(gym.Env):
    def __init__(self,id,GUI) -> None:
        super(FingerRobotOdeEnv, self).__init__()

        self.simTime = 0
        self.env_id  = 0
        self.animiationTimeStep = 0.1

        self.env_id = id
        self.GUI    = GUI
        self.episode_number = 0
        self.reset()
        

        if (self.GUI):            
            self.fig = plt.figure()            
            self.ax  = self.fig.add_subplot(projection='3d')
            self.ax.set_xlim(-0.08, 0.08)
            self.ax.set_ylim(-0.06, 0.06)
            self.ax.set_zlim(-0.0, 0.06)
                  
            self.ax.set_xlabel('x (m)')
            self.ax.set_ylabel('y (m)')
            self.ax.set_zlabel('z (m)')

            # self.ax.legend()
            
        
        self.action_space = spaces.Box(low=np.array([-1,-1,-1]), high=np.array([1,1,1]), dtype="float32")

        observation_bound = np.array([np.inf,      np.inf,     np.inf,  np.inf,
                                    np.inf,      np.inf,     np.inf,  np.inf,
                                    np.inf,      np.inf,     np.inf,  np.inf])
                                    
        self.observation_space = spaces.Box(low = -observation_bound, high = observation_bound, dtype="float32")
                

    def observe(self):
        ob      = np.concatenate([self.states])
        return ob


    def step(self, action):
        self.u  = action

        # % total length
        self.l  = self.l0 + action[0] 
        
        # if action[0] == 0:       
        #     self.ds     =  0.0001
        # else:    
        #     self.ds     =  action[0]
        
            
           
        self.uy = (action[1]) / (self.l*self.d)
        self.ux = (action[2]) / -(self.l*self.d)

        self.lastStates = np.copy(self.states)
        # self.states = self.odeStep()
        self.states = self.odeStepFull()        
        self.up     = self.u

        if (self.GUI):
            self.ax.scatter (self.states[0],self.states[1],self.states[2], c = 'g', label='robot')
            line = np.vstack((self.lastStates,self.states))
            self.ax.plot3D(line[:,0], line[:,1],line[:,2], c='r',lw=2)
            plt.pause(self.animiationTimeStep)
            # if (self.simTime==0):
            #     input("press a key")


        self.reward = 1

        terminal      = self.simCableLength > 0.08
        self.simTime += 1
        

        if (self.env_id == 0 ): # only env_0 is able to print data  
            print (f"rew:{self.reward}")

        if (self.GUI and terminal):
            plt.show()


        observation = self.observe()
        info = {"rew":self.reward}
        return observation, 0.01*self.reward, terminal, info


    def reset(self):
        self.simTime = 0
        self.simCableLength   = 0
        self.episode_number  += 1

        # initial length of robot
        self.l0 = 70e-3
        # cables offset
        self.d  = 7.5e-3
        # ode step
        self.ds     = 0.0001  

        r0 = np.array([0,0,0]).reshape(3,1)  
        R0 = np.eye(3,3)
        R0 = np.reshape(R0,(9,1))
        y0 = np.concatenate((r0, R0), axis=0)
        self.states = np.squeeze(np.asarray(y0))
        self.y0 = np.copy(self.states)

        self.u      = np.array([0,0,0])
        self.tfinal = 10
        
        self.ref    = np.array([0, 0, 0])
        
        if (self.env_id == 0): #Test env
            print ("reset Env 0")
    
        observation = self.observe()
        
        return observation  # reward, done, info can't be included

    def close (self):
        print ("Environment is closing....")


    def FingerTest(self,q,nTimeStep):
        self.reset()
        # initial length of robot
        l0 = 100e-3
        # cables offset
        d  = 7.5e-3
        # % total length
        l  = l0 + q[0]

        self.uy = (q[1]) / -(l*d)
        self.ux = (q[2]) / (l*d)
        
        # %% Solving ode for shape
        s_span = np.array([0,l])

        r0 = np.array([0,0,0]).reshape(3,1)  
        R0 = np.eye(3,3)
        R0 = np.reshape(R0,(9,1))
        y0 = np.concatenate((r0, R0), axis=0)
        self.states = np.squeeze(np.asarray(y0))

        self.simCableLength = 0
        state = np.copy(self.states)
        for i in range(nTimeStep):
            state = np.vstack((state,self.odeStep()))

        self.visualize(state)    

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

    def odeStep(self):        
        cableLength          = (self.simCableLength,self.simCableLength + self.ds)
        t_eval               = np.array([cableLength[1]]) #np.linspace(t0, tfinal, int(tfinal/ts))
        sol                  = solve_ivp(self.odeFunction,cableLength,self.states,t_eval=t_eval)
        self.simCableLength += self.ds
        self.states          = np.squeeze(np.asarray(sol.y))
        #print ("t: {0}, states: {1}".format(self.sim_time,self.states))
        return self.states
    
    def odeStepFull(self):        
        cableLength          = (0,self.l)
        t_eval               = np.linspace(0, self.l, int(self.l/self.ds))
        sol                  = solve_ivp(self.odeFunction,cableLength,self.y0,t_eval=t_eval)
        self.simCableLength += self.ds
        self.states          = np.squeeze(np.asarray(sol.y[:,-1]))
        #print ("t: {0}, states: {1}".format(self.sim_time,self.states))
        return self.states
    
    def visualize(self,state):
        fig = plt.figure()
        ax  = fig.add_subplot(projection='3d')
        ax.scatter (state[:,0],state[:,1],state[:,2], c = 'g', label='robot')
        ax.plot3D(state[:,0], state[:,1],state[:,2], c='r',lw=2)

        
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')
        ax.legend()
        
        plt.show()

    
def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = FingerRobotOdeEnv(env_id,GUI=False)
        #DummyVecEnv([lambda: CustomEnv()]) #gym.make(env_id)
        env.seed(seed + rank)
        return env
    # set_global_seeds(seed)
    set_random_seed(seed)
    return _init

if __name__ =="__main__":
    """
      TODO_LIST:
    """        

    # simple test        
    # FREnv = FingerRobotOdeEnv(id=0,GUI=True)
    # q = np.array([0.01,-0.01,0.01])
    # FREnv.FingerTest(q=q, nTimeStep=10)
    
    # create environment
    num_cpu_core = 1
    max_epc = 100

    if (num_cpu_core == 1):
        env = FingerRobotOdeEnv(id = 0,GUI=True)
    else:
        env = SubprocVecEnv([make_env(i, i) for i in range(1, num_cpu_core)]) # Create the vectorized environment
    
    # Test enviroment with a fixed q for 100 steps
    env.reset()   
    q = np.array([0.01,-0.01,-0.01])
    states = np.copy(env.states)
    for i in range(100):
        obs, rewards, done, info = env.step(q)
        if done:
            break
        states = np.vstack((states,obs[0:12]))
    
    # Training using PPO   
    timestr   = time.strftime("%Y%m%d-%H%M%S")
    modelName = "learnedPolicies/model_"+ timestr
    logFname  = "learnedPolicies/log_"  + timestr
   
    model = PPO("MlpPolicy", env,verbose=0,tensorboard_log=logFname)    
    model.learn(total_timesteps = max_epc)
   
    print ("saving the learned policy")
    model.save(modelName)
    del model
    env.close()
    


   






