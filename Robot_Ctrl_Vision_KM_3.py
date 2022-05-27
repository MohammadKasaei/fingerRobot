
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

# from VerifyingODEModel import FingerRobotOdeEnv
from NumJac import SoftRobotControl


def mainFunc():

  vision = visionSys()
  robot = FingerRobot(DoControl=True)
  

  # env = SoftRobotControl()
  # q = np.array([0.0,-0.0,-0.0])
  
  # xdot = np.array((-0.0,-0.0,0.0))
  
  # robot.sendMessage ([0,0,0,0]) 
  # time.sleep(10)

  # ts = 0.04
  # logState = np.array([])
  # for i in range(25*3):    
  #     jac = env.Jac(env.runOdeForJac,q).T
  #     pseudo_inverse = np.linalg.pinv(jac)
  #     qdot = pseudo_inverse @ xdot        
  #     q   += (qdot * ts)
         
  #     r1 =  q[0]+ q[1]
  #     r2 =  q[0]- q[1]
  #     r3 =  q[0]+ q[2]
  #     r4 =  q[0]- q[2]
      
  #     robot.sendMessage ([int(r1*1000),int(r2*1000),int(r3*1000),int(r4*1000)]) 
      
  #     # plt.plot(i*ts,r1,'r*')
  #     # plt.plot(i*ts,r2,'ro')
  #     # plt.plot(i*ts,r3,'b*')
  #     # plt.plot(i*ts,r4,'bo')
      
      
  #     # plt.plot(i*ts,q[0],'r*')
  #     # plt.plot(i*ts,q[1],'go')
  #     # plt.plot(i*ts,q[2],'bx')
      
  #     # plt.plot(i*ts,env.states[0],'r*')
  #     # plt.plot(i*ts,env.states[1],'g*')
  #     # plt.plot(i*ts,env.states[2],'b*')
      
  #     # if i==0:
  #     #     logState = np.copy(env.states)
  #     #     plt.grid()
  #     # else:
  #     #     logState =  np.vstack((logState,env.states))
      

  #     # plt.pause(ts)


  # plt.show()  

  env = SoftRobotControl()
  q = np.array([0.0,-0.0,-0.0])

  xdot = np.array((-0.0,-0.0,0.00))
  ts = 0.045

  while (False):        
    key = vision.update(vis=True,freq=10,waitForKey=50)
    # print (key)
    if (key==27):
        break         
    xc = vision.targetPose[0]
    print (f"pos: {xc[0]:3.3f} \t{xc[1]:3.3f} \t{xc[2]:3.3f} \t ")
    
    robot.updateKeyboardCtrl(key)        
    robot.robotCtrlTeleOp(robot.lengthKey,robot.phiKey,robot.thetaKey)
  
  robot.sendMessage ([0,0,0,0]) 
  time.sleep(10)
  
  logState = np.array([])
  tp = time.time() 
  t0 = tp
  
  logState = np.array([])
  
  r1 = 0
  r2 = 0
  r3 = 0
  r4 = 0
  
  r1p = 0
  r2p = 0
  r3p = 0
  r4p = 0
  
  xd = np.array([0.008,0.033,0.101]).reshape(3,1)
  xdot = np.array((0.0,0.0,0.0))
  
  fig = plt.figure()     
  i = 0
    
  logState  = np.array([])
  preview   = True
  logging   = True
  plotting  = False
  doControl = False
  
  K = np.diag((.45,.45,.45))   

  timestr = time.strftime("%Y%m%d-%H%M%S")
  logFname  = "logs/log_"  + timestr+".dat"

  # print ("============================================================")

  # for i in range(20):
  #   r1 = i*0.0005
  #   robot.sendMessage ([(r1*1000),(r1*1000),(r1*1000),(r1*1000)])

  while (True):        
        t = time.time()
        gt = t - t0
        dt = t - tp
        tp = t
        
        key = vision.update(vis=preview,freq=15)
        if (key==27):
           break   
         
        xc = vision.targetPose[0]
        print (f"dt:{dt:3.3f}\t pos: {xc[0]:3.3f} \t{xc[1]:3.3f} \t{xc[2]:3.3f} \t ")
        
        # if robot.inputMode == 0: # teleoperation
        #   robot.updateKeyboardCtrl(key)        
        #   robot.robotfigureCtrlTeleOp(robot.lengthKey,robot.phiKey,robot.thetaKey)
        
        if (gt>5):

          ################################  Circle and Helix ##########################
          # T  = 70.0
          # w  = 2*np.pi/T
          # x0 = np.array((0.005,0.025,0.126))

          # xd = (x0 + np.array((0.03*np.sin(w*(gt-5)),0.03*np.cos(w*(gt-5)),(0.0/150.0)*(gt-5)))).reshape(3,1)
          # xdot = np.array((0.03*w*np.cos(w*(gt-5)),-0.03*w*np.sin(w*(gt-5)),(0.0/150.0)))
          

          ################################  Square #####################################
          # T  = 20.0
          # x0 = np.array((-0.01,0.028,0.10))
          # tt = (gt-5)

          # if (tt<T):
          #   xd = (x0 + 2*np.array((-0.01+(0.02/T)*tt,0.01,0.01))).reshape(3,1)
          #   xdot = 2*np.array(((0.02/T),0,0))            
          # elif (tt<2*T):
          #   xd = (x0 + 2*np.array((0.01,0.01-((0.02/T)*(tt-T)),0.01))).reshape(3,1)
          #   xdot = 2*np.array((0,-(0.02/T),0))            
          # elif (tt<3*T):
          #   xd = (x0 + 2*np.array((0.01-((0.02/T)*(tt-(2*T))),-0.01,0.01))).reshape(3,1)
          #   xdot = 2*np.array((-(0.02/T),0,0))            
          # elif (tt<4*T):
          #   xd = (x0 + 2*np.array((-0.01,-0.01+((0.02/T)*(tt-(3*T))),0.01))).reshape(3,1)
          #   xdot = 2*np.array((0,+(0.02/T),0))     
          # else:
          #   t0 = time.time()+5
          #####################################################################

          
          jac = env.Jac(env.runOdeForJac,q).T
          pseudo_inverse = np.linalg.pinv(jac)
          xc = xc[0:3].reshape(3,1)
          qdot = pseudo_inverse @ (xdot + np.squeeze((K@(xd-xc)).T) )
          
          
          # if (gt>8):
          #   xdot = np.array((-0.0,-0.0,0.0))
              
                
          q   += (qdot * ts)
          r1 =  q[0]+ q[1]
          r2 =  q[0]- q[1]
          r3 =  q[0]+ q[2]
          r4 =  q[0]- q[2]
          
          
          if doControl:
            robot.sendMessage ([(r1*1000),(r2*1000),(r3*1000),(r4*1000)]) 

          r1p = r1
          r2p = r2
          r3p = r3
          r4p = r4

          if (logging):   
            dummyLog = np.concatenate((np.array((gt,dt)),np.squeeze(xc),np.squeeze(xd),np.squeeze(xdot),np.squeeze(qdot),np.array((q[0],q[1],q[2])),np.squeeze(vision.targetPose),np.squeeze(vision.targetPoseRaw)))
            if logState.shape[0] == 0:
              logState = np.copy(dummyLog)
            else:  
              logState = np.vstack((logState,dummyLog))

            if i%25 == 0:  
              np.savetxt(logFname,logState)
            

          if i==0:          
              # logState = np.copy(dummyLog)
              if (plotting):
                plt.grid()

          # else:
          #     logState = np.vstack((logState,dummyLog))
          #     np.savetxt(logFname,logState)
            
              
              # with open("test.txt", "ab") as f:
              #   np.savetxt(f, logState,delimiter=',')
              #   f.write(b"\n")
              


          if (plotting and i%20==0): 
            plt.plot(gt,xd[0],'k+')          
            plt.plot(gt,xd[1],'k+')          
            plt.plot(gt,xd[2],'k+')          
            
            plt.plot(gt,xc[0],'r*')          
            plt.plot(gt,xc[1],'go')
            plt.plot(gt,xc[2],'b*')
            plt.pause(0.001)
            
            # if (int(gt)%10 == 0):
            #     plt.clf()
            #     plt.grid()
                
            if (i==0):           
              plt.grid()
            # i = 1
            
            
            
          i+=1
            
          # plt.plot(i*ts,r4,'bo')
          
            
            # plt.plot(i*ts,q[0],'r*')
            # plt.plot(i*ts,q[1],'go')
            # plt.plot(i*ts,q[2],'bx')
            
            # plt.plot(i*ts,env.states[0],'r*')
            # plt.plot(i*ts,env.states[1],'g*')
            # plt.plot(i*ts,env.states[2],'b*')
            
            # if i==0:
            #     logState = np.copy(env.states)
            #     plt.grid()
            # else:
            #     logState =  np.vstack((logState,env.states))
            

            # plt.pause(ts)


        # plt.show()  

          
          
          


  while (True):
        
        deltaTime = time.time() - start_time
        
        key = vision.update(vis=True,freq=20)
        
        
        vision.markerPose[1]
        
        
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
  
  