#   https://www.kvaser.com/linux-drivers-and-sdk-2/

from cProfile import label
from locale import MON_1, MON_2
from operator import length_hint
# from turtle import pos
import numpy as np
from numpy import dtype
import pybullet as p
from scipy.spatial.transform import Rotation as R
from numpy import argsort                    
import math 
import can
from can.interface import Bus
import struct
import ctypes
import time

import scipy.integrate as integrate        

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

from visionSys import visionSys

#Finger Robot Class
class FingerRobot():
    
    def __init__(self,DoControl=True):
        # set up connection to hardware
        self.printEnable = False
        
        can.rc['interface'] = "kvaser"
        can.rc['channel'] = '0'
        can.rc['bitrate'] = 500000
        self.bus = Bus()
        
        self.lengthOffset = 10  # length offset
        self.angleTune = 6      # angle tuning coef (experimentally!)
        self.inputMode = 0      # 0: Tele-Operation & 1: Vision-Based
        
        self.DoControl = DoControl
    
        self.lengthKey = 0
        self.thetaKey = 0
        self.phiKey = 0
        
        self.phiL = 0
        self.thetaL = 0
        self.d1 = 0
        self.theta0 = 0
        self.phi0 = 0
        self.m = 0
        self.n = 0
        
        self.l1 = 0
        self.l2 = 0
        self.deltaL = 0
        
        self.lengthMarker = 0
        self.thetaMarker = 0
        self.phiMarker = 0
        
        self.PzMarkerTarget = 0
        self.PyMarkerTarget = 0
        self.PxMarkerTarget = 0
        
        self.allMotorVal = 0
        self.phiCtrl_M1_M2 = 0
        self.thetaCtrl_M3_M4 = 0
        
        self.M1 = 0
        self.M2 = 0
        self.M3 = 0
        self.M4 = 0
        self.Mvec = np.zeros((4,1))
        
        self.endTip_Pos = np.zeros((3,1))
        # self.endTip_desPos = np.zeros((3,1))
        # self.endTip_PosError = np.zeros((3,1))
        self.endTip_desPos = np.zeros(3)
        self.endTip_PosError = np.zeros(3)
        
        self.markerPose = np.zeros(3)
        
        self.Jac = np.zeros((3,3))
        self.JacT = np.zeros((3,3))
        self.pseudoJac = np.zeros((3,3))
        
        # self.Jac = np.zeros((3,4))
        # self.JacT = np.zeros((4,3))
        # self.pseudoJac = np.zeros((4,3))

        self.K = np.eye(3,3)
        self.K[0,0] = 2.5
        self.K[1,1] = 2.5
        self.K[2,2] = 2.5
        
        

        self.bus = can.Bus(channel='0', bustype="kvaser", bitrate=500000)
        
        self.t0 = time.time()
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
        
        if not self.DoControl:
            return
        
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
            # time.sleep(0.0005)
            
            
            # toggle new position
            msg = can.Message(arbitration_id=0x401 + i,
                            data=[0x0F, 0x00, 0x00, 0x00, 0x00, 0x00],
                            is_extended_id=False)
            self.bus.send(msg)            
            #time.sleep(0.01)
            time.sleep(0.005)
        
    def robotCtrlTeleOp(self,length,phi,theta):
                
        allMotorVal = length    #+ lengthCtrlMarker_New  # length control addition/subtraction from length offset
        phiCtrl_M1_M2 = phi/self.angleTune       # theta control (Z-Y plane)
        thetaCtrl_M3_M4 = theta/self.angleTune   # phi control (Z-X plane)    
        
        M1 = allMotorVal+phiCtrl_M1_M2      # phi ==> rotation around y-axis (Z-X plane)
        M2 = allMotorVal-phiCtrl_M1_M2      # phi ==> rotation around y-axis (Z-X plane)
        M3 = allMotorVal+thetaCtrl_M3_M4    # theta ==> rotation around x-axis (Z-Y plane)
        M4 = allMotorVal-thetaCtrl_M3_M4    # theta ==> rotation around x-axis (Z-Y plane)
        
        totalLength = self.lengthOffset + allMotorVal # total length of the robot
        
        endTip_PosX = (totalLength * math.cos((phi-90) * (math.pi/180)))
        endTip_PosY = (totalLength * math.cos((theta-90) * (math.pi/180))) 
        endTip_PosZ = ((totalLength/2) * (math.sin((theta+90) * (math.pi/180)) + math.sin((phi+90) * (math.pi/180))))
        self.endTip_Pos = [endTip_PosX/1000, endTip_PosY/1000, endTip_PosZ/1000]
        
        self.sendMessage ([M1,M2,M3,M4])
    
    def robotIK(self,length,phi,theta):

        alpha = 0.1
        beta  = 0.1
        length = np.clip (length,0,0.05)
        phi    = np.clip (phi,-1.309,1.309)
        theta  = np.clip (theta,-1.309,1.309)
        
        self.allMotorVal     =  ((1-alpha) * self.allMotorVal   )  + (alpha * (length*1000))  #+ lengthCtrlMarker_New  # length control addition/subtraction from length offset
        self.phiCtrl_M1_M2   =  ((1-beta)  * self.phiCtrl_M1_M2 )  + (beta * (math.degrees(phi)/self.angleTune))      # theta control (Z-Y plane)
        self.thetaCtrl_M3_M4 =  ((1-beta)  * self.thetaCtrl_M3_M4) + (beta * (math.degrees(theta)/self.angleTune))  # phi control (Z-X plane)    
        
        self.M1 = int(self.allMotorVal+self.phiCtrl_M1_M2)      # phi ==> rotation around y-axis (Z-X plane)
        self.M2 = int(self.allMotorVal-self.phiCtrl_M1_M2)      # phi ==> rotation around y-axis (Z-X plane)
        self.M3 = int(self.allMotorVal+self.thetaCtrl_M3_M4)    # theta ==> rotation around x-axis (Z-Y plane)
        self.M4 = int(self.allMotorVal-self.thetaCtrl_M3_M4)    # theta ==> rotation around x-axis (Z-Y plane)
        
        totalLength = self.lengthOffset + self.allMotorVal # total length of the robot
        
        endTip_PosX = (totalLength * math.cos((phi-1.5708)))
        endTip_PosY = (totalLength * math.cos((theta-1.5708))) 
        endTip_PosZ = ((totalLength/2) * (math.sin((theta+1.5708)) + math.sin((phi+1.5708))))
        self.endTip_Pos = [endTip_PosX/1000, endTip_PosY/1000, endTip_PosZ/1000]
        
        if (time.time()-self.t0 > 5 ):  
            self.sendMessage ([self.M1,self.M2,self.M3,self.M4])
                    
    def robotFK(self, M1, M2, M3, M4):
        
        #Initialization
        self.theta0 = (math.pi)/2
        r      = 7.5 #0.0075
        beta   = (math.pi)/2
        
        # self.M1 = 0
        # self.M2 = 0
        # self.M3 = 0
        # self.M4 = 0
             
        # L and Li (i = 1,2,3,4)    
        # L1 = self.lengthOffset + self.M2
        # L2 = self.lengthOffset + self.M4
        # L3 = self.lengthOffset + self.M1
        # L4 = self.lengthOffset + self.M3 
        
        L1 = self.lengthOffset + M2
        L2 = self.lengthOffset + M4
        L3 = self.lengthOffset + M1
        L4 = self.lengthOffset + M3        
                
        L = (L1 + L2 + L3 + L4)/4
        
        # phiL
        A = [(L - L1)*math.cos(beta) - (L - L2)]
        B = [(L - L1)*math.sin(beta)]
        
        # phiL condition 
        if (M1 > M2 or M3 > M4):
            self.phiL = np.arctan2(A,B);
        else:
            self.phiL = (np.arctan2(A,B) + math.pi);
        ##############################################    

        # phiL no condition
        # self.phiL = np.arctan2(A,B)
        ##############################################

        # theta L
        phi1 = self.phiL
        phi2 = self.phiL + (math.pi)/2
        phi3 = self.phiL + (math.pi)
        phi4 = self.phiL + 3*(math.pi)/2
        
        self.d1 = r*math.cos(0)
        d2 = r*math.cos(phi2)
        d3 = r*math.cos(phi3)
        d4 = r*math.cos(phi4)
        
        # thetaL condition
        if (L - L1 == 0):
            self.thetaL = (self.theta0 - (L-L2)/self.d1)
        elif (L - L2 == 0):
            self.thetaL = (self.theta0 - (L-L1)/self.d1)
        elif (L - L1 > 0 and L - L2 < 0):
            self.thetaL = (self.theta0 - (L-L1)/self.d1)
        elif (L - L1 < 0 and L - L2 > 0):
            self.thetaL = (self.theta0 - (L-L2)/self.d1)
        else:
            self.thetaL = (self.theta0 - (L-L1)/self.d1)
        ##############################################
        
        # thetaL no condition
        # self.thetaL = (self.theta0 - (L-L1)/self.d1)
        # self.thetaL = (self.theta0 - (2*L-(L1+L2))/self.d1)
        ##############################################
        
        #rotation matrices:
        c_z = math.cos(-self.phiL)
        s_z = math.sin(-self.phiL)
        Rb1_z = np.array(((c_z, -s_z, 0), (s_z, c_z, 0), (0, 0, 1)))
        
        c_y = math.cos((L-L1)/self.d1)
        s_y = math.sin((L-L1)/self.d1)
        R1e_y = np.array(((c_y, 0, s_y), (0, 1, 0), (-s_y, 0, c_y)))
        
        Rbe = Rb1_z @ R1e_y

        posZX = np.array([L*np.cos(self.thetaL),0,L*np.sin(self.thetaL)])
     
        posE = Rb1_z @ posZX
        print ("\nL: ", L)
        print ("L1: ", L1)
        print ("L2: ", L2)
        print ("L3: ", L3)
        print ("L4: ", L4)
        print("thetaL = ", self.thetaL, "\tphiL = ", self.phiL)
        print ("EndTipPos = ", f"{posE[0]:3.4f} \t{posE[1]:3.4f} \t{posE[2]:3.4f}")

        return posE , self.phiL
    
    def robotFKnew(self, l1,l2,deltaL):
        
        #Initialization
        self.theta0 = (math.pi)/2
        self.phi0 = (math.pi)/2;
        r      = 7.5 #0.0075
        beta   = (math.pi)/2
        
        # self.M1 = 0
        # self.M2 = 0
        # self.M3 = 0
        # self.M4 = 0
             
        # L and Li (i = 1,2,3,4)    
        # L1 = self.lengthOffset + self.M2
        # L2 = self.lengthOffset + self.M4
        # L3 = self.lengthOffset + self.M1
        # L4 = self.lengthOffset + self.M3 
        
        M1 = deltaL + l1
        M2 = deltaL - l1
        M3 = deltaL + l2
        M4 = deltaL - l2
           
        L1 = self.lengthOffset + M2
        L2 = self.lengthOffset + M4
        L3 = self.lengthOffset + M1
        L4 = self.lengthOffset + M3        
                
        L = (L1 + L2 + L3 + L4)/4
         
        self.d1 = r*math.cos(0)
        self.thetaL = (self.theta0 - (L-L1)/self.d1)
        self.phiL = (self.phi0 - (L-L2)/self.d1)
        
        # phiL condition
        # A = [(L - L1)*math.cos(beta) - (L - L2)]
        # B = [(L - L1)*math.sin(beta)]
        
        # if (M1 > M2 or M3 > M4):
        #     self.phiL = np.arctan2(A,B)[0];
        # else:
        #     self.phiL = (np.arctan2(A,B) + math.pi)[0];
        ##############################################
        
        # phiL no condition
        # self.phiL = np.arctan2(A,B)
        ##############################################

        # theta L
        # phi1 = self.phiL
        # phi2 = self.phiL + (math.pi)/2
        # phi3 = self.phiL + (math.pi)
        # phi4 = self.phiL + 3*(math.pi)/2
       
        # d2 = r*math.cos(phi2)
        # d3 = r*math.cos(phi3)
        # d4 = r*math.cos(phi4)
        
        # thetaL condition - Modified
        # if (M2 != 0 or M4 != 0) and (M2 == 0 or M4 == 0):
        #     if abs(M1) < abs(M3):
        #         self.m = 0;
        #         self.n = 1;
        #         cond = 'thetaL Cond 1'
        #     elif abs(M1) > abs(M3):
        #         self.m = 1;
        #         self.n = 0;
        #         cond = 'thetaL Cond 2'
        #     else:
        #         self.m = 1;
        #         self.n = 1;
        #         cond = 'thetaL Cond 3'

        #     self.thetaL = self.theta0 - ((self.m+self.n)*L - (self.m*L1+self.n*L2))/self.d1
        # else:
        #     if M2 > 0 and M4 < 0:
        #         self.m = 1;
        #         self.n = 0;
        #         cond = 'thetaL Cond 4'
        #     elif M2 < 0 and M4 > 0:
        #         self.m = 0;
        #         self.n = 1;
        #         cond = 'thetaL Cond 5'
        #     elif (abs(M2) > abs(M4)) or (abs(M2) < abs(M4)):
        #         self.m = 1;
        #         self.n = 1;
        #         cond = 'thetaL Cond 6'
        #     else:
        #         self.m = 1/2;
        #         self.n = 1/2;
        #         cond = 'thetaL Cond 7'
                
        #     self.thetaL = self.theta0 - ((self.m+self.n)*L - (self.m*(L1)+self.n*(L2)))/self.d1
        
        # thetaL condition
        # if (L - L1 == 0):
        #     self.thetaL = (self.theta0 - (L-L2)/self.d1)
        # elif (L - L2 == 0):
        #     self.thetaL = (self.theta0 - (L-L1)/self.d1)
        # elif (L - L1 > 0 and L - L2 < 0):
        #     self.thetaL = (self.theta0 - (L-L1)/self.d1)
        # elif (L - L1 < 0 and L - L2 > 0):
        #     self.thetaL = (self.theta0 - (L-L2)/self.d1)
        # else:
        #     self.thetaL = (self.theta0 - (L-L1)/self.d1)
        ############################################## 
            
        # thetaL no codntion    
        # self.thetaL = (self.theta0 - (L-L1)/self.d1)
        # self.thetaL = (self.theta0 - (2*L-(L1+L2))/self.d1)
        ##############################################
        
        #rotation matrices:
        # c_z = math.cos(-self.phiL)
        # s_z = math.sin(-self.phiL)
        # Rb1_z = np.array(((c_z, -s_z, 0), (s_z, c_z, 0), (0, 0, 1)))
        
        # c_y = math.cos((L-L1)/self.d1)
        # s_y = math.sin((L-L1)/self.d1)
        # R1e_y = np.array(((c_y, 0, s_y), (0, 1, 0), (-s_y, 0, c_y)))
        
        # Rbe = Rb1_z @ R1e_y

        posZX = np.array([L*np.cos(self.thetaL),0,L*np.sin(self.thetaL)])
     
        # posEnew = Rb1_z @ posZX
        
        posEnew = [L*np.cos(self.thetaL),L*np.cos(self.phiL),L*(np.sin(self.thetaL)*np.sin(self.phiL))]
        
        if (self.printEnable):
            print ("\nL: ", L)
            print ("L1: ", L1)
            print ("L2: ", L2)
            print ("L3: ", L3)
            print ("L4: ", L4)
            print("thetaL = ", self.thetaL, "\tphiL = ", self.phiL)
            print ("EndTipPos = ", f"{posEnew[0]:3.4f} \t{posEnew[1]:3.4f} \t{posEnew[2]:3.4f}")

        return posEnew , self.phiL
    
    
    def visualizeFK(self):
        
        num = 10
        
        M2 = np.linspace(0,10,num)
        M4 = np.linspace(0,10,num) 
        M1 = np.linspace(0,10,num)
        M3 = np.linspace(0,10,num) 

        color = np.linspace(0,1,num)
    
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
       
        for i in range(num):
            pose = self.robotFK(M1[i],M2[i],M3[i],M4[i])[0]
            # pose = [i,i,i]
            ax.scatter(pose[0],pose[1],pose[2],color = [0,0,color[i]])        
            # plt.hold(True)
            if (self.printEnable):
                print("pose:",pose)
            
            
        ax.set(xlim=(-10, 10),  xticks=np.arange(-10, 10),
                ylim=(-10, 10), yticks=np.arange(-10, 10),
                zlim=(0, 20), zticks=np.arange(0, 20))

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()  
        
        if (self.printEnable):
             print("End Pose:", self.endTip_Pos)
        # time.sleep(1)
        
    def visualizeFKnew(self):
        
        # fig = plt.figure() 
        # M = 1000
        # angle = np.exp(1j * 2 * np.pi / M)
        # angles = np.cumprod(np.ones(M + 1) * angle)
        # l1, l2 = 5 * np.real(angles), 5 * np.imag(angles)
        # plt.plot(l1, l2)
            
        num = 10                
        l1 = np.linspace(0,10,num)
        l2 = np.linspace(0,10,num)
        deltaL = np.linspace(0,0,num) 
        
        color = np.linspace(0,1,num)
    
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
       
        for i in range(num):
            pose = self.robotFKnew(l1[i],l2[i],deltaL[i])[0]
            # pose = [i,i,i]
            ax.scatter(pose[0],pose[1],pose[2],color = [0,0,color[i]])        
            # plt.hold(True)
        
            if (self.printEnable):
                 print("pose:",pose)
            
            
        ax.set(xlim=(-10, 10),  xticks=np.arange(-10, 10),
                ylim=(-10, 10), yticks=np.arange(-10, 10),
                zlim=(0, 20), zticks=np.arange(0, 20))

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()  
        
        if (self.printEnable):
             print("End Pose:", self.endTip_Pos)
        # time.sleep(1)
    
    
    def visualizeFKnewCricle(self):
        
        # fig = plt.figure() 
        # M = 1000
        # angle = np.exp(1j * 2 * np.pi / M)
        # angles = np.cumprod(np.ones(M + 1) * angle)
        # l1, l2 = 5 * np.real(angles), 5 * np.imag(angles)
        # plt.plot(l1, l2)
            
        num = 20         
        
        x = np.linspace(-10,10,num)
        y = np.sqrt(10**2-x**2)
        
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # color = np.linspace(0,1,num)
        # ax.scatter(x,y,0*x+10,color = [0,0,1])   
        # plt.show()    
    
        
        l1 = x #np.linspace(0,10,num)
        l2 = y #np.linspace(0,10,num)
        
        deltaL = np.linspace(0,0,num) 
        
        color = np.linspace(0,1,num)
    
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
       
        for i in range(num):
            pose = self.robotFKnew(l1[i],l2[i],deltaL[i])[0]
            # pose = [i,i,i]
            ax.scatter(pose[0],pose[1],pose[2],color = [0,0,color[i]])        
            # plt.hold(True)
        
            if (self.printEnable):
                 print("pose:",pose)
            
            
        ax.set(xlim=(-10, 10),  xticks=np.arange(-10, 10),
                ylim=(-10, 10), yticks=np.arange(-10, 10),
                zlim=(0, 20), zticks=np.arange(0, 20))

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()  
        
        if (self.printEnable):
             print("End Pose:", self.endTip_Pos)
        # time.sleep(1)
        
    def visualizePD(self):
        
        # pose0 = self.robotFK(0,0,0,0)[0]
        
        pose0 = self.robotFKnew(0,0,10)[0]
        
        desPosX = 0
        desPosY = 0
        desPosZ = 0
       
        endTip_desPos = np.zeros(3)
        endTip_desPos[0] = pose0[0] + desPosX
        endTip_desPos[1] = pose0[1] + desPosY
        endTip_desPos[2] = pose0[2] + desPosZ
        
        self.robotCtrlPD(endTip_desPos - pose0)
        
        # self.Mvec[0] = 0
        # self.Mvec[1] = 0
        # self.Mvec[2] = 10 
        # self.Mvec[3] = -10 
        
        # trajectory 
          
        num = 10 
        t = np.linspace(0.0,10,num)  #np.zeros(num)
        M1 = np.linspace(0,self.Mvec[0],num)  #np.zeros(num)
        M2 = np.linspace(0,self.Mvec[1],num)  #np.zeros(num)
        M3 = np.linspace(0,self.Mvec[2],num) 
        M4 = np.linspace(0,self.Mvec[3],num)
        
        # M1 = 0.001* np.linspace(0,10,num)  #np.zeros(num)
        # M2 = 0.001* -np.linspace(0,10,num)  #np.zeros(num)
        # M3 = -np.linspace(0,10,num) 
        # M4 = np.linspace(0,10,num)
           
        color  = np.linspace(0,1,num)
        color2 = np.linspace(0,1,num)
    
        fig = plt.figure()      
        plt.plot(t,M1,'r',ls='solid',label='M1')
        plt.plot(t,M2,'g',ls='dotted',label='M2')
        plt.plot(t,M3,'b',ls='dashed',label='M3')
        plt.plot(t,M4,'k',ls='dashdot',label='M4')
        plt.title (f"xd={ endTip_desPos[0]:3.2f} yd={ endTip_desPos[1]:3.2f} zd={ endTip_desPos[2]:3.2f} \n m1 = {self.Mvec[0]:3.2f}, m2 = {self.Mvec[1]:3.2f}, m3 = {self.Mvec[2]:3.2f}, m4 = {self.Mvec[3]:3.2f}")
        plt.xlabel("Time(s)")
        plt.ylabel("pos(mm)")
        plt.grid()
        plt.legend()
        
        fig = plt.figure()  
        plt.grid()  
        ax = Axes3D(fig)

        self.robotFK(M1[0],M2[0],M3[0],M4[0])[0]            
        
        self.M1 = M1[0]
        self.M2 = M2[0]
        self.M3 = M3[0]
        self.M4 = M4[0]
               
        for i in range(num):
            
            # pose = self.robotFK(M1[i],M2[i],M3[i],M4[i])[0]            
            endTip_desPos = self.robotFK(M1[i],M2[i],M3[i],M4[i])[0]            
            
            CurentPos = self.robotFK(self.M1,self.M2,self.M3,self.M4)[0]                      
            ax.scatter(endTip_desPos[0],endTip_desPos[1],endTip_desPos[2],color = [0,0,color[i]],label = 'desPose')   
            
            self.robotCtrlPD(endTip_desPos - CurentPos)
             
            self.M1 += (self.Mvec[0])
            self.M2 += (self.Mvec[1])
            self.M3 += (self.Mvec[2])
            self.M4 += (self.Mvec[3])
            
            poseCtrl = self.robotFK(self.M1,self.M2,self.M3,self.M4)[0]
            
            ax.scatter(poseCtrl[0],poseCtrl[1],poseCtrl[2],color = [color2[i],0,0],label = 'Ctrl')    
            if (self.printEnable):
                 print ("Motor Values: ", int(self.M1), int(self.M2), int(self.M3), int(self.M4))
            
        #plt.legend()   
        # ax.set(xlim=(-20, 20), xticks=np.arange(-20, 20),
        #        ylim=(-20, 20), yticks=np.arange(-20, 20),
        #        zlim=(0, 20),   zticks=np.arange(0, 20))
        
       
        
        ax.set(xlim=(-20, 20), 
               ylim =(-20, 20), 
               zlim =(0, 20))

        ax.view_init(elev=50, azim=270)
            
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()  
    
    
    def visualizePD2(self):
        
        # pose0 = self.robotFK(0,0,0,0)[0]
        
        pose0 = self.robotFKnew(0,0,10)[0]
        
        desPosX = 0
        desPosY = 0
        desPosZ = 0
       
        endTip_desPos = np.zeros(3)
        endTip_desPos[0] = pose0[0] + desPosX
        endTip_desPos[1] = pose0[1] + desPosY
        endTip_desPos[2] = pose0[2] + desPosZ
        
        num = 10 
                
        self.l1 = np.linspace(0,10,num)  #np.zeros(num)
        self.l2 = np.linspace(0,0,num)  #np.zeros(num)
        self.deltaL = np.linspace(0,0,num) 
         
        self.ll1 = self.l1[0]
        self.ll2 = self.l2[0]
        self.ddeltaL = self.deltaL[0]

        self.robotCtrlPD(endTip_desPos - pose0)
        
        # self.M1 = self.Mvec[0]
        # self.M2 = self.Mvec[1]
        # self.M3 = self.Mvec[2]
        # self.M4 = self.Mvec[3]
        
        # trajectory 
          
        # num = 10 
        # t = np.linspace(0.0,10,num)  #np.zeros(num)
        # M1 = np.linspace(0,self.Mvec[0],num)  #np.zeros(num)
        # M2 = np.linspace(0,self.Mvec[1],num)  #np.zeros(num)
        # M3 = np.linspace(0,self.Mvec[2],num) 
        # M4 = np.linspace(0,self.Mvec[3],num)
        
        # M1 = 0.001* np.linspace(0,10,num)  #np.zeros(num)
        # M2 = 0.001* -np.linspace(0,10,num)  #np.zeros(num)
        # M3 = -np.linspace(0,10,num) 
        # M4 = np.linspace(0,10,num)
           
        color  = np.linspace(0,1,num)
        color2 = np.linspace(0,1,num)
    
        # fig = plt.figure()      
        # plt.plot(t,M1,'r',ls='solid',label='M1')
        # plt.plot(t,M2,'g',ls='dotted',label='M2')
        # plt.plot(t,M3,'b',ls='dashed',label='M3')
        # plt.plot(t,M4,'k',ls='dashdot',label='M4')
        # plt.title (f"xd={ endTip_desPos[0]:3.2f} yd={ endTip_desPos[1]:3.2f} zd={ endTip_desPos[2]:3.2f} \n m1 = {self.Mvec[0]:3.2f}, m2 = {self.Mvec[1]:3.2f}, m3 = {self.Mvec[2]:3.2f}, m4 = {self.Mvec[3]:3.2f}")
        # plt.xlabel("Time(s)")
        # plt.ylabel("pos(mm)")
        # plt.grid()
        # plt.legend()
        
        fig = plt.figure()  
        plt.grid()  
        ax = Axes3D(fig)

        
        # l1 = np.linspace(0,0,num)  #np.zeros(num)
        # l2 = np.linspace(0,0,num)  #np.zeros(num)
        # deltaL = np.linspace(0,10,num) 
        err = []     
               
        for i in range(num):
            
            # pose = self.robotFK(M1[i],M2[i],M3[i],M4[i])[0]            
            # l1 = 0
            # l2 = 0
            # deltaL = 10
            
            endTip_desPos = self.robotFKnew(self.l1[i],self.l2[i],self.deltaL[i])[0]            
            
            dlc = (self.M1+self.M2)/2
            l1c = self.M1 - dlc
            l2c = self.M3 - dlc
        
            
            CurentPos = self.robotFKnew(l1c,l2c,dlc)[0]                      
            ax.scatter(endTip_desPos[0],endTip_desPos[1],endTip_desPos[2],color = [0,0,color[i]],label = 'desPose')   
            
            err.append(np.array(endTip_desPos) - np.array(CurentPos))
            
            self.ll1 = self.l1[i]
            self.ll2 = self.l2[i]
            self.ddeltaL = self.deltaL[i]
            
            self.robotCtrlPD(np.array (endTip_desPos) - np.array(CurentPos))
             
            self.M1 += self.L1L2DL[2]  + self.L1L2DL[0] 
            self.M2 += self.L1L2DL[2]  - self.L1L2DL[0]
            self.M3 += self.L1L2DL[2]  + self.L1L2DL[1]
            self.M4 += self.L1L2DL[2]  - self.L1L2DL[1]
            
            # poseCtrl = self.robotFK(self.M1,self.M2,self.M3,self.M4)[0]
            
            # dlct = (self.M1+self.M2)/2
            # l1ct = self.M1 - dlct
            # l2ct = self.M3 - dlct
            
            dlct = self.L1L2DL[2]            
            l1ct = self.L1L2DL[0]            
            l2ct = self.L1L2DL[1]
            
            
            
            poseCtrl = self.robotFKnew(l1ct,l2ct,dlct)[0]    
            
            ax.scatter(poseCtrl[0],poseCtrl[1],poseCtrl[2],color = [color2[i],0,0],label = 'Ctrl')    
            if (self.printEnable):
                 print ("Motor Values: ", int(self.M1), int(self.M2), int(self.M3), int(self.M4))
            
        #plt.legend()   
        # ax.set(xlim=(-20, 20), xticks=np.arange(-20, 20),
        #        ylim=(-20, 20), yticks=np.arange(-20, 20),
        #        zlim=(0, 20),   zticks=np.arange(0, 20))
        
       
        
        ax.set(xlim=(-20, 20), 
               ylim =(-20, 20), 
               zlim =(0, 20))

        ax.view_init(elev=50, azim=270)
            
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.title("red = ctrl blue = des")
        plt.show()  
        
        fig = plt.figure()  
        plt.plot(np.array(err)[:,0],'r-o',label = "x")
        plt.plot(np.array(err)[:,1],'g-o',label = "y")
        plt.plot(np.array(err)[:,2],'b-o',label = "z")
        plt.grid()
        plt.legend()
        
        plt.show()
        
        
                
    def robotCtrlPD(self, endTip_PosError):
 
        self.endTip_PosError = endTip_PosError
        
        # self.Jac[0][0] = (math.cos(self.theta0 + (self.m*(self.M2 + self.lengthOffset) - (self.m + self.n)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset) + self.n*(self.M4 + self.lengthOffset))/self.d1)*math.cos(self.phiL))/4 + (math.sin(self.theta0 + (self.m*(self.M2 + self.lengthOffset) - (self.m + self.n)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset) + self.n*(self.M4 + self.lengthOffset))/self.d1)*math.cos(self.phiL)*(self.m/4 + self.n/4)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset))/self.d1
        # self.Jac[0][1] = (math.cos(self.theta0 + (self.m*(self.M2 + self.lengthOffset) - (self.m + self.n)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset) + self.n*(self.M4 + self.lengthOffset))/self.d1)*math.cos(self.phiL))/4 - (math.sin(self.theta0 + (self.m*(self.M2 + self.lengthOffset) - (self.m + self.n)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset) + self.n*(self.M4 + self.lengthOffset))/self.d1)*math.cos(self.phiL)*((3*self.m)/4 - self.n/4)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset))/self.d1
        # self.Jac[0][2] = (math.cos(self.theta0 + (self.m*(self.M2 + self.lengthOffset) - (self.m + self.n)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset) + self.n*(self.M4 + self.lengthOffset))/self.d1)*math.cos(self.phiL))/4 + (math.sin(self.theta0 + (self.m*(self.M2 + self.lengthOffset) - (self.m + self.n)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset) + self.n*(self.M4 + self.lengthOffset))/self.d1)*math.cos(self.phiL)*(self.m/4 + self.n/4)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset))/self.d1
        # self.Jac[0][3] = (math.cos(self.theta0 + (self.m*(self.M2 + self.lengthOffset) - (self.m + self.n)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset) + self.n*(self.M4 + self.lengthOffset))/self.d1)*math.cos(self.phiL))/4 + (math.sin(self.theta0 + (self.m*(self.M2 + self.lengthOffset) - (self.m + self.n)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset) + self.n*(self.M4 + self.lengthOffset))/self.d1)*math.cos(self.phiL)*(self.m/4 - (3*self.n)/4)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset))/self.d1
 
        # self.Jac[1][0] = (math.cos(self.theta0 + (self.m*(self.M2 + self.lengthOffset) - (self.m + self.n)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset) + self.n*(self.M4 + self.lengthOffset))/self.d1)*math.sin(self.phiL))/4 + (math.sin(self.theta0 + (self.m*(self.M2 + self.lengthOffset) - (self.m + self.n)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset) + self.n*(self.M4 + self.lengthOffset))/self.d1)*math.sin(self.phiL)*(self.m/4 + self.n/4)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset))/self.d1
        # self.Jac[1][1] = (math.cos(self.theta0 + (self.m*(self.M2 + self.lengthOffset) - (self.m + self.n)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset) + self.n*(self.M4 + self.lengthOffset))/self.d1)*math.sin(self.phiL))/4 - (math.sin(self.theta0 + (self.m*(self.M2 + self.lengthOffset) - (self.m + self.n)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset) + self.n*(self.M4 + self.lengthOffset))/self.d1)*math.sin(self.phiL)*((3*self.m)/4 - self.n/4)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset))/self.d1
        # self.Jac[1][2] = (math.cos(self.theta0 + (self.m*(self.M2 + self.lengthOffset) - (self.m + self.n)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset) + self.n*(self.M4 + self.lengthOffset))/self.d1)*math.sin(self.phiL))/4 + (math.sin(self.theta0 + (self.m*(self.M2 + self.lengthOffset) - (self.m + self.n)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset) + self.n*(self.M4 + self.lengthOffset))/self.d1)*math.sin(self.phiL)*(self.m/4 + self.n/4)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset))/self.d1
        # self.Jac[1][3] = (math.cos(self.theta0 + (self.m*(self.M2 + self.lengthOffset) - (self.m + self.n)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset) + self.n*(self.M4 + self.lengthOffset))/self.d1)*math.sin(self.phiL))/4 + (math.sin(self.theta0 + (self.m*(self.M2 + self.lengthOffset) - (self.m + self.n)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset) + self.n*(self.M4 + self.lengthOffset))/self.d1)*math.sin(self.phiL)*(self.m/4 - (3*self.n)/4)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset))/self.d1
        
        # self.Jac[2][0] = math.sin(self.theta0 + (self.m*(self.M2 + self.lengthOffset) - (self.m + self.n)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset) + self.n*(self.M4 + self.lengthOffset))/self.d1)/4 - (math.cos(self.theta0 + (self.m*(self.M2 + self.lengthOffset) - (self.m + self.n)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset) + self.n*(self.M4 + self.lengthOffset))/self.d1)*(self.m/4 + self.n/4)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset))/self.d1
        # self.Jac[2][1] = math.sin(self.theta0 + (self.m*(self.M2 + self.lengthOffset) - (self.m + self.n)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset) + self.n*(self.M4 + self.lengthOffset))/self.d1)/4 + (math.cos(self.theta0 + (self.m*(self.M2 + self.lengthOffset) - (self.m + self.n)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset) + self.n*(self.M4 + self.lengthOffset))/self.d1)*((3*self.m)/4 - self.n/4)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset))/self.d1
        # self.Jac[2][2] = math.sin(self.theta0 + (self.m*(self.M2 + self.lengthOffset) - (self.m + self.n)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset) + self.n*(self.M4 + self.lengthOffset))/self.d1)/4 - (math.cos(self.theta0 + (self.m*(self.M2 + self.lengthOffset) - (self.m + self.n)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset) + self.n*(self.M4 + self.lengthOffset))/self.d1)*(self.m/4 + self.n/4)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset))/self.d1
        # self.Jac[2][3] = math.sin(self.theta0 + (self.m*(self.M2 + self.lengthOffset) - (self.m + self.n)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset) + self.n*(self.M4 + self.lengthOffset))/self.d1)/4 - (math.cos(self.theta0 + (self.m*(self.M2 + self.lengthOffset) - (self.m + self.n)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset) + self.n*(self.M4 + self.lengthOffset))/self.d1)*(self.m/4 - (3*self.n)/4)*(self.M1/4 + self.M2/4 + self.M3/4 + self.M4/4 + self.lengthOffset))/self.d1
    
        self.Jac[0][0] = (math.sin(self.theta0 - self.ll1/self.d1)*(self.ddeltaL + self.lengthOffset))/self.d1
        self.Jac[0][1] = 0
        self.Jac[0][2] = math.cos(self.theta0 - self.ll1/self.d1)

        self.Jac[1][0] = 0
        self.Jac[1][1] = (math.sin(self.phi0 - self.ll2/self.d1)*(self.ddeltaL + self.lengthOffset))/self.d1
        self.Jac[1][2] = math.cos(self.phi0 - self.ll2/self.d1)
        
        self.Jac[2][0] = -(math.cos(self.theta0 - self.ll1/self.d1)*(self.ddeltaL + self.lengthOffset))/self.d1
        self.Jac[2][1] = -(math.cos(self.phi0 - self.ll2/self.d1)*(self.ddeltaL + self.lengthOffset))/self.d1
        self.Jac[2][2] = math.sin(self.phi0 - self.ll2/self.d1) + math.sin(self.theta0 - self.ll1/self.d1)
        # print("Jac: ", self.Jac)
        
        self.JacT = np.transpose(self.Jac)
        
        self.pseudoJac = np.linalg.pinv(self.Jac)
       
        # self.Mvec   = self.pseudoJac @ (self.endTip_PosError)
        
        self.L1L2DL = self.pseudoJac @ self.K @ (self.endTip_PosError)
        
        
        
        if (self.printEnable):
            print("Mvec: ", self.Mvec)
        # self.M1 = int(self.Mvec[0])
        # self.M2 = int(self.Mvec[1])
        # self.M3 = int(self.Mvec[2])
        # self.M4 = int(self.Mvec[3])
        
        # if (time.time()-self.t0 > 5 ):  
        #     self.sendMessage ([self.M1,self.M2,self.M3,self.M4])
            
        # print ("Motor Values: ", self.M1,self.M2,self.M3,self.M4)
           
    def robotFKvision(self,l1,l2,deltaL):
        
        M1 = deltaL+l1   # phi ==> rotation around y-axis (Z-X plane)
        M2 = deltaL-l1   # phi ==> rotation around y-axis (Z-X plane)
        M3 = deltaL+l2   # theta ==> rotation around x-axis (Z-Y plane)
        M4 = deltaL-l2   # theta ==> rotation around x-axis (Z-Y plane)
        
        self.sendMessage ([M1,M2,M3,M4]) 
         
        # self.markerPose = vision.getMarkerPose(0) 
        # return self.markerPose
          
    def updateKeyboardCtrl(self,key):
        
        if key == 27:
            self.lengthKey = 0
            self.thetaKey = 0
            self.phiKey = 0
    
        elif key == 171:
            self.lengthKey += 1
            if self.lengthKey >= 50:
                self.lengthKey = 50
        
        elif key == 173:
            self.lengthKey -=1
            if self.lengthKey <= 0:
                self.lengthKey = 0

        elif key == 184:
            self.thetaKey += 1
            if self.thetaKey >= 90:
                self.thetaKey = 90

        elif key == 178:
            self.thetaKey -= 1
            if self.thetaKey <= - 90:
                self.thetaKey = -90

        elif key == 182:
            self.phiKey += 1
            if self.phiKey >= 90:
               self.phiKey = 90

        elif key == 180:
            self.phiKey -=1
            if self.phiKey <= -90:
              self.phiKey = -90
              

                      
    def updateMarkerBasedCtrl2(self,targetPose):
        
        self.lengthMarker = np.linalg.norm (targetPose[0][0:3]) - (self.lengthOffset/1000.0)
        self.phiMarker   = np.arctan2(targetPose[0][0],targetPose[0][2])
        self.thetaMarker = np.arctan2(targetPose[0][1],targetPose[0][2])
        
        self.PzMarkerTarget = (((targetPose[0][2]))*(math.sin(math.radians(self.thetaMarker+90)) + math.sin(math.radians(self.phiMarker+90))))/2
        self.PyMarkerTarget = targetPose[0][1] + 0.036
        self.PxMarkerTarget = targetPose[0][0]

        return self.lengthMarker, self.thetaMarker, self.phiMarker
        
    def updateMarkerBasedCtrl(self,targetPose):

        self.PzMarkerTarget = (((targetPose[0][2]))*(math.sin(math.radians(self.thetaMarker+90)) + math.sin(math.radians(self.phiMarker+90))))/2
        self.lengthMarker   = np.clip(int(self.PzMarkerTarget*1000 - self.lengthOffset),1,75)
        
        self.PyMarkerTarget = targetPose[0][1] + 0.036
        tmp_y = self.lengthOffset + self.lengthMarker

        if tmp_y == 0:
            tmp_y = 0.001

        tmp_y_thetaCtrl = np.clip(self.PyMarkerTarget*1000/(tmp_y),-1,1)
        self.thetaMarker = -(((math.acos(tmp_y_thetaCtrl)) * 180/math.pi) - 90)
        
        self.PxMarkerTarget = targetPose[0][0]
        tmp_x = self.lengthOffset + self.lengthMarker
        
        if tmp_x == 0:
            tmp_x = 0.001
            
        tmp_x_phiCtrl = np.clip(self.PxMarkerTarget*1000/(tmp_x),-1,1)
        self.phiMarker = -(((math.acos(tmp_x_phiCtrl)) * 180/math.pi) - 90)
                    
        return self.lengthMarker, self.thetaMarker, self.phiMarker
    
    def printFunc(self):
        if self.inputMode == 0:
            print("\nRobot Length Ctrl (mm) = ", self.lengthKey)
            print("Theta Value Ctrl (Z-Y plane) = ", self.thetaKey)
            print("Phi Value Ctrl (Z-X plane) = ", self.phiKey)
            print('End Tip Position (m) = ', f"{self.endTip_Pos[0]:3.3f} \t {self.endTip_Pos[1]:3.3f} \t {self.endTip_Pos[2]:3.3f}")
            
        elif self.inputMode == 1:
            print("Px_Target_marker = ", f"{self.PxMarkerTarget:3.3f}", "\tPy_Target_marker = ", f"{self.PyMarkerTarget:3.3f}", "\tPz_Target_marker = ", f"{self.PzMarkerTarget:3.3f}")
            print("\nlengthMarker (mm) = ", self.lengthMarker)
            print("thetaMarker (deg)", math.degrees(self.thetaMarker))
            print("phiMarker (deg)", math.degrees(self.phiMarker))
            print('End Tip Position (m) = ', f"{self.endTip_Pos[0]:3.3f} \t {self.endTip_Pos[1]:3.3f} \t {self.endTip_Pos[2]:3.3f}")
        elif self.inputMode == 2:
            pass
            # print("Px_Target_marker = ", f"{self.PxMarkerTarget:3.3f}", "\tPy_Target_marker = ", f"{self.PyMarkerTarget:3.3f}", "\tPz_Target_marker = ", f"{self.PzMarkerTarget:3.3f}")
            # print("\nlengthMarker (mm) = ", self.lengthMarker)
            # print("thetaMarker (deg)", math.degrees(self.thetaMarker))
            # print("phiMarker (deg)", math.degrees(self.phiMarker))
            # print('End Tip Position (m) = ', f"{self.endTip_Pos[0]:3.3f} \t {self.endTip_Pos[1]:3.3f} \t {self.endTip_Pos[2]:3.3f}")
            