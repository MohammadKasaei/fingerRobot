from __future__ import print_function
from inspect import ArgSpec
import sys # Python 2/3 compatibility
import cv2                            
import numpy as np
import pybullet as p

from scipy.spatial.transform import Rotation as R
from numpy import argsort                    
import math 
import can
from can.interface import Bus
import struct
import ctypes
import time

#Global Variables
global lengthOffset, allMotorVal
global angleTune
global M1, M2, M3, M4
global thetaCtrl_M3_M4, theta
global phiCtrl_M1_M2, phi
global endTip_PosX, endTip_PosY, endTip_PosZ
global endTip_Pos
global deltaTime

#global point, x, y
global point, x, y

#Finger Robot Class
class FingerRobot():
    
    def __init__(self):
        # set up connection to hardware
        can.rc['interface'] = "kvaser"
        can.rc['channel'] = '0'
        can.rc['bitrate'] = 500000
        self.bus = Bus()
        
        self.bus = can.Bus(channel='0', bustype="kvaser", bitrate=500000)
    
        self.clearFaults()
        self.startRemoteNode()
        self.enableAllMotors()
        self.setAllMotorModes()
        self.setMaxMotorVelocity()

    def clearFaults(self):
        for i in range(1, 5):
            msg = can.Message(arbitration_id=0x600 + i,
                                data=[int("40", 16), int("41", 16), int("60", 16), 0, 0, 0, 0, 0],
                                is_extended_id=False)
            self.bus.send(msg)
    
    def startRemoteNode(self):
        # Start remote node via NMT
        #  different commands can be used to set operation mode (pre-op, start, etc). For all of them the
        #  Cob Id is 0 in NMT mode. Data has two bytes. First byte is a desired command, one of
        #  the five following commands can be used
        #  80 is pre-operational
        #  81 is reset node
        #  82 is reset communication
        #  01 is start
        #  02 is stop
        #  second byte is node id, can be 0 (all nodes) or a number between 1 to 256.

         for i in range(1, 5):
            msg = can.Message(arbitration_id=0x0,
                                data=[1, i],
                                is_extended_id=False)
            self.bus.send(msg)
            #time.sleep(0.01)
            time.sleep(0.005)

    def enableAllMotors(self):
        # Enable All Motors
        for i in range(1, 5):
            msg = can.Message(arbitration_id=0x200 + i,
                                data=[0x00, 0x00],
                                is_extended_id=False)                   
            self.bus.send(msg)
            #time.sleep(0.01)
            time.sleep(0.005)

        for i in range(1, 5):
            msg = can.Message(arbitration_id=0x200 + i,
                                data=[0x06, 0x00],
                                is_extended_id=False)
            self.bus.send(msg)
            #time.sleep(0.01)
            time.sleep(0.005)

        for i in range(1, 5):
            msg = can.Message(arbitration_id=0x200 + i,
                                data=[0x0F, 0x00],
                                is_extended_id=False)
            self.bus.send(msg)
            #time.sleep(0.01)
            time.sleep(0.005)

    def setAllMotorModes(self,mode = 0):
        if mode == 0:
            for i in range(1, 5):
                msg = can.Message(arbitration_id=0x300 + i,
                                    data=[0x0F, 0x00, 0x01],
                                    is_extended_id=False)
                self.bus.send(msg)
        else:
            pass

    def setMaxMotorVelocity(self,maxRPM=0):
        # Set rotational speed of motors
        for i in range(1, 5):
            msg = can.Message(arbitration_id=0x600 + i,
                                data=[0x22, 0x81, 0x60, 0x0, 0x40, 0x1F, 0x0, 0x0], # 5000 rpm
                                is_extended_id=False)
            self.bus.send(msg)

    # functions for swapping bytes and turning position data into hexadecimal
    def pos2message(self,i):
        num = ctypes.c_uint32(i).value  # convert int into uint
        num2 = struct.unpack("<I", struct.pack(">I", num))[0]  # swap bytes
        output = hex(num2)[2:]
        return output.zfill(8)

    def sendMessage(self,data):
       
        for i in range(0, len(data)):
            Data = self.pos2message(int(data[i] * (-612459.2 / (2 * 3.8))))
            # set position value
            msg = can.Message(arbitration_id=0x401 + i,
                            data=[0x3F, 0x00, int(Data[0:2], 16), int(Data[2:4], 16), int(Data[4:6], 16),
                                    int(Data[6:8], 16)],
                            is_extended_id=False)
            self.bus.send(msg)
            #time.sleep(0.01)
            time.sleep(0.005)
            
            # toggle new position
            msg = can.Message(arbitration_id=0x401 + i,
                            data=[0x0F, 0x00, 0x00, 0x00, 0x00, 0x00],
                            is_extended_id=False)
            self.bus.send(msg)            
            #time.sleep(0.01)
            time.sleep(0.005)

# Dictionary that was used to generate the ArUco marker
aruco_dictionary_name = "DICT_ARUCO_ORIGINAL"
 
# The different ArUco dictionaries built into the OpenCV library. 
ARUCO_DICT = {
  "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
  "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
  "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
  "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
  "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
  "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
  "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
  "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
  "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
  "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
  "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
  "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
  "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
  "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
  "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
  "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
  "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}
 
# Side length of the ArUco marker in meters 
aruco_marker_side_length = 0.0785
 
# Calibration parameters yaml file
camera_calibration_parameters_filename = 'calibration_chessboard.yaml'
 
def euler_from_quaternion(x, y, z, w):
  """
  Convert a quaternion into euler angles (roll, pitch, yaw)
  roll is rotation around x in radians (counterclockwise)
  pitch is rotation around y in radians (counterclockwise)
  yaw is rotation around z in radians (counterclockwise)
  """
  t0 = +2.0 * (w * x + y * z)
  t1 = +1.0 - 2.0 * (x * x + y * y)
  roll_x = math.atan2(t0, t1)
      
  t2 = +2.0 * (w * y - z * x)
  t2 = +1.0 if t2 > +1.0 else t2
  t2 = -1.0 if t2 < -1.0 else t2
  pitch_y = math.asin(t2)
      
  t3 = +2.0 * (w * z + x * y)
  t4 = +1.0 - 2.0 * (y * y + z * z)
  yaw_z = math.atan2(t3, t4)
      
  return roll_x, pitch_y, yaw_z # in radians
 
if __name__ == "__main__":

    start_time = time.time()    
    robot = FingerRobot()
    lengthOffset = 77   # length offset
    angleTune = 6       # angle tuning coef (experimentally!)

    ######################## vision part ########################
    # Check that we have a valid ArUco marker
    if ARUCO_DICT.get(aruco_dictionary_name, None) is None:
        print("[INFO] ArUCo tag of '{}' is not supported".format(
        args["type"]))
        sys.exit(0)
 
    # Load the camera parameters from the saved file
    cv_file = cv2.FileStorage(
    camera_calibration_parameters_filename, cv2.FILE_STORAGE_READ) 
    mtx = cv_file.getNode('K').mat()
    dst = cv_file.getNode('D').mat()
    cv_file.release()
     
    # Load the ArUco dictionary
    print("[INFO] detecting '{}' markers...".format(
    aruco_dictionary_name))
    this_aruco_dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_dictionary_name])
    this_aruco_parameters = cv2.aruco.DetectorParameters_create()
   
    # Start the video stream
    cap = cv2.VideoCapture(2)
    #############################################################

    while True:
        deltaTime = time.time() - start_time

    ######################## control part ########################
        phi = 0     #22.5*math.sin(0.5*deltaTime)
        theta = 0   #22.5*math.sin(0.5*deltaTime)
            
        allMotorVal = 0  #15*math.sin(0.5*deltaTime)   # length control addition/subtraction from length offset
        phiCtrl_M1_M2 = phi/angleTune       # theta control (Z-Y plane)
        thetaCtrl_M3_M4 = theta/angleTune   # phi control (Z-X plane)
        
        M1 = allMotorVal+phiCtrl_M1_M2      # phi ==> rotation around y-axis (Z-X plane)
        M2 = allMotorVal-phiCtrl_M1_M2      # phi ==> rotation around y-axis (Z-X plane)
        M3 = allMotorVal+thetaCtrl_M3_M4    # theta ==> rotation around x-axis (Z-Y plane)
        M4 = allMotorVal-thetaCtrl_M3_M4    # theta ==> rotation around x-axis (Z-Y plane)

        totalLength = lengthOffset + allMotorVal # total length of the robot

    # End-Tip Position Calculation
        endTip_PosX = (totalLength * math.cos((phi-90) * (math.pi/180)))
        endTip_PosY = (totalLength * math.cos((theta-90) * (math.pi/180))) 
        endTip_PosZ = ((totalLength/2) * (math.sin((theta+90) * (math.pi/180)) + math.sin((phi+90) * (math.pi/180))))
        endTip_Pos = [endTip_PosX/10, endTip_PosY/10, endTip_PosZ/10]


        robot.sendMessage ([M1,M2,M3,M4])

        print('\nPhi = ', phi, ', Phi Ctrl = +/-', phiCtrl_M1_M2, ', Theta = ', theta, ', Theta Ctrl = +/-', thetaCtrl_M3_M4)
        print('\nM1 (cm) = ', M1/10, ', M2 (cm) = ', M2/10, ', M3 (cm) = ', M3/10, ', M4 (cm) = ', M4/10)
        print('\nTotal Length (cm) = ', totalLength/10)
        print('\nEnd Tip Position (cm) = ', endTip_Pos)
        print('\nTime = ', deltaTime)

robot.bus.shutdown()