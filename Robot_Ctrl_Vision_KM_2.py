
from __future__ import print_function
from locale import DAY_1
from re import L, M
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
from numpy import argsort                    
from can.interface import Bus
import time
from FingerRobot import FingerRobot
from visionSys import visionSys

from testODE import SFODE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

from VerifyingODEModel import FingerRobotOdeEnv



def mainFunc():

  vision = visionSys()
  robot = FingerRobot(DoControl=True)
  
  
  env = FingerRobotOdeEnv(id = 0,GUI=True)
  
  # Test enviroment with a fixed q for 100 steps
  r1 = 0
  r2 = 0
  r3 = 0
  r4 = 0
    
  env.reset()     
  states = np.copy(env.states)
  dl = 0.0
  l1 = -0.00
  l2 = -0.00
  
  r1 = dl
  r2 = dl
  r3 = dl
  r4 = dl
  
  
  
  for i in range(10):
        
        dl = 0.002
        l1 = 0.001
        l2 = 0.00
        
        
        r1 +=  dl+l1
        r2 +=  dl-l1
        r3 +=  dl+l2
        r4 +=  dl-l2
        
        q = np.array([(i+1)*dl,(i+1)*l1,(i+1)*l2])
        
        robot.sendMessage ([int(r1*1000),int(r2*1000),int(r3*1000),int(r4*1000)]) 
        
        obs, rewards, done, info = env.step(q)
        print (f"x{obs[0]:3.3f}\t y{obs[1]:3.3f}\t z{obs[2]:3.3f}")
        input("press a key")
        if done:
            break
        states = np.vstack((states,obs[0:12]))
        
        
      
        
  
  robot.x_offset = vision.x_offset
  robot.y_offset = vision.y_offset
  robot.z_offset = vision.z_offset
  
  
  # t0 = time.time()
  # for i in range(10):
  #   robot.sendMessage ([0,0,0,0])     
  # print ("================================ time: ")
  # print(time.time()-t0)
  
  
        
  
  # 0: Tele-Operation Ctrl 
  # 1: Vision-Based Ctrl
  # 2: Vision-Based FK - PD Ctrl

  desPose = np.zeros(3)
  #curPose = np.zeros(3)
  
  robot.inputMode = 0
      
  lastCommand = time.time()
  
  l1 = 0
  l2 = 0
  deltaL = 0
  
  # robot.visualizeFK()
  # robot.visualizeFKnew()
  # robot.visualizeFKnewCricle()
  # robot.visualizePD2()
  # sys = SFODE()
  # x0 =  np.array([0,0,0,0,0,0])    
  # sys.states = np.copy(x0)
  # y = [np.array(x0)]    
  # sys.ref = np.array([0.05,0.01,0.05,0.01, 0.,0.])
  # sys.sim_time = 0
  # tfinal = 10
  
  # u = [np.array([0,0,0])]
  
  # st = np.array([0])
  # rrr = [np.array([0,0,0,0])]

  # stp = 0
  
  # for _ in range (int(tfinal/sys.ts)):   
  #     # sys.ref = 0.05*np.array([np.sin(sys.sim_time),np.cos(sys.sim_time),1-np.cos(sys.sim_time),np.sin(sys.sim_time), sys.sim_time,1])     
  #     yn = sys.singleODEStep()
  #     y = np.append(y,[yn],axis=0)
  #     u = np.append(u,sys.u,axis=0)
  
  #     stc = np.sqrt(yn[1]**2 + yn[3]**2 + yn[5]**2)
      
  #     st = np.append (st,[stc+stp],axis=0)
  #     stp = stc
      
  #     delta = 0.1 #0.0075

  #     r1 = stc+np.array([yn[1],yn[3],yn[5]]).dot(np.array([delta*np.cos(0*np.pi/2), delta*np.sin(0*np.pi/2),0 ]))
  #     r2 = stc+np.array([yn[1],yn[3],yn[5]]).dot(np.array([delta*np.cos(1*np.pi/2), delta*np.sin(1*np.pi/2),0 ]))
  #     r3 = stc+np.array([yn[1],yn[3],yn[5]]).dot(np.array([delta*np.cos(2*np.pi/2), delta*np.sin(2*np.pi/2),0 ]))
  #     r4 = stc+np.array([yn[1],yn[3],yn[5]]).dot(np.array([delta*np.cos(3*np.pi/2), delta*np.sin(3*np.pi/2),0 ]))

  #     rrr = np.append (rrr,[np.array([r1,r2,r3,r4])],axis=0)

  

  # fig, axs = plt.subplots(4)
  
  # t = np.linspace(0,tfinal,y.shape[0])
  
  # axs[0].plot(t,y[:,0],'ro',label='x')    
  # axs[0].plot(t,y[:,1],'r',label='xdot')
  # axs[0].plot(t,np.ones(len(t))*0.05,'k--',lw=3,label='xref')
  # axs[0].plot(t,0.1*u[:,0],'c--',lw=3,label='0.1*u_x')
  
  

  # axs[0].legend()
  # axs[0].grid()
  
  
  # axs[1].plot(t,y[:,2],'go',label='y')
  # axs[1].plot(t,y[:,3],'g--',label='ydot')
  # axs[1].plot(t,np.ones(len(t))*0.05,'k--',lw=3,label='yref')
  # axs[1].plot(t,0.1*u[:,1],'c--',lw=3,label='0.1*u_y')

  
  # axs[1].legend()
  # axs[1].grid()
  
  # axs[2].plot(t,y[:,4],'bo',label='z')
  # axs[2].plot(t,y[:,5],'b--',label='zdot')
  # axs[2].plot(t,np.ones(len(t))*0.0,'k--',lw=3,label='zref')
  # axs[2].plot(t,0.1*u[:,2],'c--',lw=3,label='0.1*u_z')
  

  # axs[2].legend()
  # axs[2].grid()
  
  
  # axs[3].plot(t,st,'bo',label='st')
  # axs[3].plot(t,rrr[:,0],'r-o',label='r1')
  # axs[3].plot(t,rrr[:,1],'g-x',label='r2')
  # axs[3].plot(t,rrr[:,2],'b-*',label='r3')
  # axs[3].plot(t,rrr[:,3],'k-o',label='r4')
  
  # axs[3].legend()
  # axs[3].grid()
  
  # plt.show()  
  
    
  # fig = plt.figure()
  # ax = fig.add_subplot(projection='3d')
  # ax.scatter (y[:,0],y[:,2],y[:,4],'r',label='robot')
  # ax.scatter (np.ones(len(t))*0.05,np.ones(len(t))*0.05,np.ones(len(t))*0.0,'k',label='ref')
  
  # ax.set_xlabel('x (m)')
  # ax.set_ylabel('y (m)')
  # ax.set_zlabel('z (m)')
  # ax.legend()
  
  # plt.show()
  
  sys = SFODE()
  
  x0 =  np.array([0,0,0,0,0.1,0])    
  sys.states = np.copy(x0)
  y = [np.array(x0)]    
  u = [np.array([0,0,0])]

  sys.sim_time = 0
  tfinal = 0.5

  st = np.array([0])
  rrr = [np.array([0,0,0,0])]
  stp = 0
  stp_xz = 0
  stp_yz = 0
  
  # tf = 1
  # tfinal = tf+(sys.Np*sys.ts)
  # t0=0  
  # stc, y = sys.calcLengthBackbone(x0,t0,sys.ts,tf,sys.ref) #:(tt,sys.ref,x0)
   

  # robot.DoControl = False
  # T = 0.5
  
  # amp = [0.05,0.05,0.025]
  # for _ in range (int(tfinal/sys.ts)):   
  #     sys.ref = np.array([amp[0]*np.sin(sys.sim_time),amp[0]*np.cos(sys.sim_time),amp[1]*(1-np.cos(sys.sim_time)),amp[1]*np.sin(sys.sim_time), amp[2]*sys.sim_time,amp[2]*1])     
       
      
  #     yn = sys.singleODEStep()
  #     y = np.append(y,[yn],axis=0)
  #     u = np.append(u,sys.u,axis=0)

  #     stc = np.sqrt(yn[1]**2 + yn[3]**2 + yn[5]**2)*sys.ts
  #     stc_xz = np.sqrt(yn[1]**2 + yn[5]**2)*sys.ts      
  #     stc_yz = np.sqrt(yn[3]**2 + yn[5]**2)*sys.ts
      
  #     sBackbone = stc+stp
      
  #     sBackbone_xz = stc_xz+stp_xz
  #     sBackbone_yz = stc_yz+stp_yz
      
  #     st = np.append (st,[sBackbone],axis=0)
  #     stp = sBackbone
      
  #     stp_xz = sBackbone_xz
  #     stp_yz = sBackbone_yz
      
  #     delta = 0.0075

  #     r1 = sBackbone_xz+np.array([yn[1],yn[3],yn[5]]).dot(np.array([delta*np.cos(0*np.pi/2), delta*np.sin(0*np.pi/2),0 ]))
  #     r2 = sBackbone_yz+np.array([yn[1],yn[3],yn[5]]).dot(np.array([delta*np.cos(1*np.pi/2), delta*np.sin(1*np.pi/2),0 ]))
  #     r3 = sBackbone_xz+np.array([yn[1],yn[3],yn[5]]).dot(np.array([delta*np.cos(2*np.pi/2), delta*np.sin(2*np.pi/2),0 ]))
  #     r4 = sBackbone_yz+np.array([yn[1],yn[3],yn[5]]).dot(np.array([delta*np.cos(3*np.pi/2), delta*np.sin(3*np.pi/2),0 ]))
      
  #     robot.sendMessage ([int(r1*1000),int(r2*1000),int(r3*1000),int(r4*1000)]) 

  #     rrr = np.append (rrr,[np.array([r1,r2,r3,r4])],axis=0)
  amp = [0.08,0.08,0.05]
  T = 0.5
  omega = 2*np.pi / T
  delta = 0.0075
  
  rrr1 = 0
  rrr2 = 0
  rrr3 = 0
  rrr4 = 0
  
  # t = np.linspace(sys.sim_time,np.clip(sys.sim_time+(sys.Np*sys.ts),0,tfinal),sys.Np)
  # refx = amp[0]*t
  # refxdot = amp[0]*np.ones(sys.Np)
  # refy = amp[1]*t
  # refydot = amp[1]*np.ones(sys.Np)
  # refz = amp[2]*t
  # refzdot = amp[2]*np.ones(sys.Np)
  # sys.ref = np.vstack ([refx,refxdot,refy,refydot,refz,refzdot])
  
  

  for _ in range (int(tfinal/sys.ts)):   
      # t = np.linspace(sys.sim_time,np.clip(sys.sim_time+(sys.Np*sys.ts),0,tfinal),sys.Np)
      # sys.ref = np.vstack ([amp[0]*np.sin(omega*t),amp[0]*omega*np.cos(omega*t),amp[1]*(1-np.cos(omega*t)),amp[1]*omega*np.sin(omega*t), x0[4] + amp[2]*t, amp[2]*np.ones(sys.Np)])
      # sys.ref = np.vstack ([amp[0]*t, amp[0]*np.ones(sys.Np), amp[1]*t, amp[1]*np.ones(sys.Np), x0[4] + amp[2]*t, amp[2]*np.ones(sys.Np)])
      # sys.ref = np.vstack ([refx,refxdot,refy,refydot,refz,refzdot])
      
      
      t = np.linspace(sys.sim_time,np.clip(sys.sim_time+(sys.Np*sys.ts),0,tfinal),sys.Np)
      refx = amp[0]*t
      refxdot = amp[0]*np.ones(sys.Np)
      refy = amp[1]*t
      refydot = amp[1]*np.ones(sys.Np)
      refz = amp[2]*t
      refzdot = amp[2]*np.ones(sys.Np)
      sys.ref = np.vstack ([refx,refxdot,refy,refydot,refz,refzdot])
      

      # sys.ref = amp*np.array([np.sin(sys.sim_time),np.cos(sys.sim_time),1-np.cos(sys.sim_time),np.sin(sys.sim_time), sys.sim_time,1])             
      # sys.ref = amp*np.array([np.sin(sys.sim_time),np.cos(sys.sim_time),1-np.cos(sys.sim_time),np.sin(sys.sim_time), sys.sim_time,1])     
      yn = sys.singleODEStep()
      y = np.append(y,[yn],axis=0)
      u = np.append(u,sys.u,axis=0)


      stc = np.sqrt(yn[1]**2 + yn[3]**2 + yn[5]**2)*sys.ts
      
      rr1 = np.sqrt((yn[1]-delta*np.sin(0*np.pi/2))**2 + (yn[3]+delta*np.cos(0*np.pi/2))**2 + yn[5]**2)*sys.ts
      rr2 = np.sqrt((yn[1]-delta*np.sin(2*np.pi/2))**2 + (yn[3]+delta*np.cos(2*np.pi/2))**2 + yn[5]**2)*sys.ts
      rr3 = np.sqrt((yn[1]-delta*np.sin(1*np.pi/2))**2 + (yn[3]+delta*np.cos(1*np.pi/2))**2 + yn[5]**2)*sys.ts
      rr4 = np.sqrt((yn[1]-delta*np.sin(3*np.pi/2))**2 + (yn[3]+delta*np.cos(3*np.pi/2))**2 + yn[5]**2)*sys.ts
      
      rrr1 = rrr1 + rr1
      rrr2 = rrr2 + rr2
      rrr3 = rrr3 + rr3
      rrr4 = rrr4 + rr4
      
    

      # r1 = sBackbone_xz+np.array([yn[1],yn[3],yn[5]]).dot(np.array([delta*np.cos(0*np.pi/2), delta*np.sin(0*np.pi/2),0 ]))
      # r2 = sBackbone_yz+np.array([yn[1],yn[3],yn[5]]).dot(np.array([delta*np.cos(1*np.pi/2), delta*np.sin(1*np.pi/2),0 ]))
      # r3 = sBackbone_xz+np.array([yn[1],yn[3],yn[5]]).dot(np.array([delta*np.cos(2*np.pi/2), delta*np.sin(2*np.pi/2),0 ]))
      # r4 = sBackbone_yz+np.array([yn[1],yn[3],yn[5]]).dot(np.array([delta*np.cos(3*np.pi/2), delta*np.sin(3*np.pi/2),0 ]))
      
      # robot.sendMessage ([int(rrr3*1000),int(rrr4*1000),int(rrr1*1000),int(rrr2*1000)]) 

      rrr = np.append (rrr,[np.array([rrr1,rrr2,rrr3,rrr4])],axis=0)


  r1 = rrr1
  r2 = rrr2
  r3 = rrr3
  r4 = rrr4
  
  fig, axs = plt.subplots(4)
  
  t = np.linspace(0,tfinal,y.shape[0])
  refx = amp[0]*t
  refxdot = amp[0]*np.ones(sys.Np)
  refy = amp[1]*t
  refydot = amp[1]*np.ones(sys.Np)
  refz = amp[2]*t
  refzdot = amp[2]*np.ones(sys.Np)
  sys.ref = ([refx,refxdot,refy,refydot,refz,refzdot])
  
  
  axs[0].plot(t,y[:,0],'ro',label='x')    
  axs[0].plot(t,y[:,1],'r',label='xdot')
  axs[0].plot(t,refx,'k--',lw=3,label='xref')
  axs[0].plot(t,u[:,0],'m--',lw=3,label='ux')
  
  

  axs[0].legend()
  axs[0].grid()
  
  
  axs[1].plot(t,y[:,2],'go',label='y')
  axs[1].plot(t,y[:,3],'g--',label='ydot')
  axs[1].plot(t,refy,'k--',lw=3,label='yref')
  axs[1].plot(t,u[:,1],'m--',lw=3,label='uy')


  axs[1].legend()
  axs[1].grid()
  
  axs[2].plot(t,y[:,4],'bo',label='z')
  axs[2].plot(t,y[:,5],'b--',label='zdot')
  axs[2].plot(t,refz,'k--',lw=3,label='zref')
  axs[2].plot(t,u[:,2],'m--',lw=3,label='uz')

  axs[2].legend()
  axs[2].grid()

  # axs[3].plot(t,st,'bo',label='st')
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
  ax.scatter (refx,refy,refz,'k',label='ref')
  
  ax.set_xlabel('x (m)')
  ax.set_ylabel('y (m)')
  ax.set_zlabel('z (m)')
  ax.legend()
  
  plt.show()
  
  



  while (True):
        
        deltaTime = time.time() - start_time
        
        key = vision.update(vis=True,freq=20)
        targetPose = vision.targetPose
    
        endTip_PosError = robot.endTip_PosError
        
        robot.updateKeyboardCtrl(key)
        
        # getting marker pose
        #markerPose = vision.getMarkerPose(0)
        
        # robot.updateMarkerBasedCtrl(targetPose)
        robot.updateMarkerBasedCtrl2(targetPose)
        
        if robot.inputMode == 0: # teleoperation
          robot.robotCtrlTeleOp(robot.lengthKey,robot.phiKey,robot.thetaKey)
        elif robot.inputMode == 1:  # vision based control
          if (time.time()-lastCommand>0.01):  
            lastCommand = time.time()          
            robot.robotIK(robot.lengthMarker,robot.phiMarker,robot.thetaMarker)
        elif robot.inputMode == 2:  # vision based PD-control
          if (time.time()-lastCommand>0.01):  
            lastCommand = time.time()          
            # robot.robotCtrlPD(robot.lengthMarker,robot.phiMarker,robot.thetaMarker)
            # desPose[0] = 5
            # desPose[1] = 0
            # desPose[2] = 15
            # curPose = robot.robotFK(robot.M1,robot.M2,robot.M3,robot.M4)[0]
            # robot.robotCtrlPD(desPose-curPose)
            
            # robot.robotFKnew(l1,l2,deltaL)
            robot.robotEndTipPosVision(10,0,0)
        elif robot.inputMode == 3:  # vision based PD-control
          if (time.time()-lastCommand>0.01):  
            lastCommand = time.time()          
            
            robot.robotFKvision(0,0,0)
            pose = vision.getMarkerPose(0) 
            
            # return self.markerPose
            print(f"Marker Pose =  {pose[0]:3.3f}\t{pose[1]:3.3f}\t{pose[2]:3.3f}")
            
            
        robot.printFunc()
        #print('Time = ', f"{deltaTime:3.3f}")
        #print('targetPose = ', targetPose)

        
if __name__ == '__main__':
  start_time = time.time()    
  mainFunc()