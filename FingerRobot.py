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
            
    # def robotControl(self,mode, z,)
