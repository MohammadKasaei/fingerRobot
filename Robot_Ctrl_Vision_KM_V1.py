
from __future__ import print_function
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
from numpy import argsort                    
from can.interface import Bus
import time
from FingerRobot import FingerRobot
from visionSys import visionSys
  
def mainFunc():

  vision = visionSys()
  robot = FingerRobot()
  
  # 0: Tele-Operation Ctrl 
  # 1: Vision-Based Ctrl
  robot.inputMode = 0
    
  while (True):
        
        deltaTime = time.time() - start_time
        
        key = vision.update(vis=True,freq=20)
        targetPose = vision.targetPose
        
        robot.updateKeyboardCtrl(key)
        #robot.updateMarkerBasedCtr2(targetPose)
        
        if robot.inputMode == 0: # teleoperation
          robot.robotIK(robot.lengthKey,robot.phiKey,robot.thetaKey)
        elif robot.inputMode == 1:  # vision based control
          robot.robotIK(robot.lengthMarker,robot.phiMarker,robot.thetaMarker)

        robot.printFunc()
        print('Time = ', f"{deltaTime:3.3f}")
        
if __name__ == '__main__':
  start_time = time.time()    
  mainFunc()