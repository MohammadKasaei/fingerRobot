import numpy as np
import can
from can.interface import Bus

import time
import struct
import ctypes
import math

#Realsense SR305
import pyrealsense2 as rs
import cv2

#Global Variables
global lengthOffset, allMotorVal
global angleTune
global M1, M2, M3, M4
global thetaCtrl_M3_M4, theta
global phiCtrl_M1_M2, phi
global endTip_PosX, endTip_PosY, endTip_PosZ
global endTip_Pos
global deltaTime

global point, x, y

#Realsense SR304 Class
class depthCamera:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        self.center_pos = (0,0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()


        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        height, width = color_image.shape[:2]
        depth_image = depth_image[0:height,300:width]
        color_image = color_image[0:height,300:width]


        #object detection activation
        self.objectDetection(color_image)

        if not depth_frame or not color_frame:
            return False, None, None
        return True, depth_image, color_image

    ######################## colored object detection part ########################
    def objectDetection(self,imageFrame):
        # Convert the imageFrame in 
        # BGR(RGB color space) to 
        # HSV(hue-saturation-value)
        # color space
        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

        # Set range for red color and 
        # define mask
        #red_lower = np.array([136, 87, 111], np.uint8)
        #red_upper = np.array([180, 255, 255], np.uint8)
        #red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

        # Set range for green color and 
        # define mask
        
        green_lower = np.array([136, 87, 111], np.uint8)
        green_upper = np.array([180, 255, 255], np.uint8)
        green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

        # Set range for blue color and
        # define mask
        #blue_lower = np.array([94, 80, 2], np.uint8)
        #blue_upper = np.array([120, 255, 255], np.uint8)
        #blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
        
        # Morphological Transform, Dilation
        # for each color and bitwise_and operator
        # between imageFrame and mask determines
        # to detect only that particular color
        kernal = np.ones((5, 5), "uint8")
        
        # For red color
        #red_mask = cv2.dilate(red_mask, kernal)
        #res_red = cv2.bitwise_and(imageFrame, imageFrame, 
        #                        mask = red_mask)
        
        # For green color
        green_mask = cv2.dilate(green_mask, kernal)
        res_green = cv2.bitwise_and(imageFrame, imageFrame,
                                mask = green_mask)
        M = cv2.moments(green_mask)
        # For blue color
        #blue_mask = cv2.dilate(blue_mask, kernal)
        #res_blue = cv2.bitwise_and(imageFrame, imageFrame,
        #                        mask = blue_mask)

        # Creating contour to track red color
        # contours, hierarchy = cv2.findContours(red_mask,
        #                                     cv2.RETR_TREE,
        #                                     cv2.CHAIN_APPROX_SIMPLE)
        
        # for pic, contour in enumerate(contours):
        #     area = cv2.contourArea(contour)
        #     if(area > 300):
        #         x, y, w, h = cv2.boundingRect(contour)
        #         imageFrame = cv2.rectangle(imageFrame, (x, y), 
        #                                 (x + w, y + h), 
        #                                 (0, 0, 255), 2)
                
        #         cv2.putText(imageFrame, "Red Colour", (x, y),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        #                     (0, 0, 255))    

        # Creating contour to track green color
        contours, hierarchy = cv2.findContours(green_mask,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
        
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                self.center_pos=(int(x+w/2), int(y+h/2))
                imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                        (x + w, y + h),
                                        (0, 255, 0), 2)
                
                cv2.putText(imageFrame, "Green Colour", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1.0, (0, 255, 0))

        # Creating contour to track blue color
        # contours, hierarchy = cv2.findContours(blue_mask,
        #                                     cv2.RETR_TREE,
        #                                     cv2.CHAIN_APPROX_SIMPLE)
        # for pic, contour in enumerate(contours):
        #     area = cv2.contourArea(contour)
        #     if(area > 300):
        #         x, y, w, h = cv2.boundingRect(contour)
        #         imageFrame = cv2.rectangle(imageFrame, (x, y),
        #                                 (x + w, y + h),
        #                                 (255, 0, 0), 2)
                
        #         cv2.putText(imageFrame, "Blue Colour", (x, y),
        #                     cv2.FONT_HERSHEY_SIMPLEX,
        #                     1.0, (255, 0, 0))
    #############################################################     
    
    def release(self):
        self.pipeline.stop()


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

if __name__ == "__main__":
    start_time = time.time()    
    robot = FingerRobot()
    lengthOffset = 77   # length offset
    angleTune = 6       # angle tuning coef (experimentally!)

    ######################## vision part ########################
    # initialize intel camera
    dc = depthCamera()
    # x = 320 
    # y = 240
    
    #############################################################

    while True:
        deltaTime = time.time() - start_time

        ######################## vision part ########################
        ret, depth_frame, color_frame = dc.get_frame()
        point = dc.center_pos
        #defining a point in the image
        cv2.circle(color_frame, point, 5, (0, 0, 255),2)
        cv2.circle(depth_frame, point, 5, (0, 0, 255),2)

        distance = depth_frame[point[1], point[0]]
        print("\nDistance = ", distance)

        #depth and color frame of the image
        cv2.imshow("Depth Frame", depth_frame)
        cv2.imshow("Color Frame", color_frame)
        key = cv2.waitKey(1)
        if key == 27: # ESC key
            break

        ######################## control part ########################
        phi = 0     #22.5*math.sin(0.5*deltaTime)
        theta = 0   #22.5*math.sin(0.5*deltaTime)
            
        allMotorVal = 50  #15*math.sin(0.5*deltaTime)   # length control addition/subtraction from length offset
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