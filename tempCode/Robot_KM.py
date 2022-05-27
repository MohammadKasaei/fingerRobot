import numpy as np
import can
from can.interface import Bus

from scipy.spatial.transform import Rotation as R


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

#global point, x, y
global point, x, y, w, h

#Realsense SR304 Class
class depthCamera:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        self.load_coefficients("camCalibration.dat")

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        self.center_pos = (0,0,0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)

    def get_frame(self,detection = True):

        
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        color_frame = frames.get_color_frame()
       
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        #object detection activation
        # self.objectDetection(color_image)
        if (detection):
            self.objectDetection2(color_image,depth_frame)


        if not depth_frame or not color_frame:
            return False, None, None , None
        return True, depth_image, color_image, depth_frame

    def objectDetection2(self,imageFrame,depthFrame):
        # Convert the imageFrame in 
        # BGR(RGB color space) to 
        # HSV(hue-saturation-value)
        # color space
        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
        # Set range for red color and 
        # define mask
        red_lower = np.array([136, 87, 111], np.uint8)
        red_upper = np.array([180, 255, 255], np.uint8)
        red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

        kernal = np.ones((5, 5), "uint8")
        
        # For red color
        red_mask = cv2.dilate(red_mask, kernal)
        res_red = cv2.bitwise_and(imageFrame, imageFrame, 
                                mask = red_mask)
        contours, hierarchy = cv2.findContours(red_mask,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
        
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)

                self.center_pos=(int(x+w/2), int(y+h/2),depthFrame.get_distance(int(x+w/2), int(y+h/2)))
                
                imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                        (x + w, y + h), 
                                        (0, 0, 255), 2)
                
                # cv2.putText(imageFrame, "End Tip", (x, y),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                #             (0, 0, 255))    

                cv2.putText(imageFrame, f"x{self.center_pos[0]} y{self.center_pos[1]} z{self.center_pos[2]:2.2f}", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, .5,
                            (0, 0, 255))    

        cv2.line(imageFrame, (320,0),(320,480), (0,200,0),2)            
        cv2.line(imageFrame, (0,240),(640,240), (0,200,0),2)            
        

    #############################################################   


    def euler_from_quaternion(self,x, y, z, w):
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
        
    ######################## colored object detection part ########################
    def objectDetection(self,imageFrame):
        # Convert the imageFrame in 
        # BGR(RGB color space) to 
        # HSV(hue-saturation-value)
        # color space
        
        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

        # Set range for red color and 
        # define mask
        red_lower = np.array([136, 87, 111], np.uint8)
        red_upper = np.array([180, 180, 255], np.uint8)

        #red_lower = np.array([25, 25, 225], np.uint8)
        #red_upper = np.array([0, 0, 255], np.uint8)
        red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

        # Set range for green color and 
        # define mask
        # green_lower = np.array([25, 52, 72], np.uint8)
        # green_upper = np.array([102, 255, 255], np.uint8)

        #green_lower = np.array([225, 25, 25], np.uint8)
        #green_upper = np.array([255, 0, 0], np.uint8)
        # green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

        # Set range for blue color and
        # define mask
        # blue_lower = np.array([94, 80, 2], np.uint8)
        # blue_upper = np.array([120, 255, 255], np.uint8)

        #blue_lower = np.array([25, 225, 25], np.uint8)
        #blue_upper = np.array([0, 255, 0], np.uint8)
        # blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
        
        # Morphological Transform, Dilation
        # for each color and bitwise_and operator
        # between imageFrame and mask determines
        # to detect only that particular color
        kernal = np.ones((5, 5), "uint8")
        
        # For red color
        red_mask = cv2.dilate(red_mask, kernal)
        res_red = cv2.bitwise_and(imageFrame, imageFrame, 
                                mask = red_mask)
        
        # For green color
        # green_mask = cv2.dilate(green_mask, kernal)
        # res_green = cv2.bitwise_and(imageFrame, imageFrame,
        #                         mask = green_mask)
        
        # For blue color
        # blue_mask = cv2.dilate(blue_mask, kernal)
        # res_blue = cv2.bitwise_and(imageFrame, imageFrame,
        #                         mask = blue_mask)

        # Creating contour to track red color
        contours, hierarchy = cv2.findContours(red_mask,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
        
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                self.center_pos=(int(x+w/2), int(y+h/2))
                imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                        (x + w, y + h), 
                                        (0, 0, 255), 2)
                
                cv2.putText(imageFrame, "End Tip", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255))    

        # Creating contour to track green color
        # contours, hierarchy = cv2.findContours(green_mask,
        #                                     cv2.RETR_TREE,
        #                                     cv2.CHAIN_APPROX_SIMPLE)
        
        # for pic, contour in enumerate(contours):
        #     area = cv2.contourArea(contour)
        #     if(area > 300):
        #         x, y, w, h = cv2.boundingRect(contour)
        #         imageFrame = cv2.rectangle(imageFrame, (x, y), 
        #                                 (x + w, y + h),
        #                                 (0, 255, 0), 2)
                
        #         cv2.putText(imageFrame, "Green Colour", (x, y),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 
        #                     1.0, (0, 255, 0))

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

    

    def load_coefficients(self,path):
        """ Loads camera matrix and distortion coefficients. """
        cv_file = cv2.FileStorage('calibration_chessboard.yaml', cv2.FILE_STORAGE_READ) 
        self.mtx = cv_file.getNode('K').mat()
        self.dst = cv_file.getNode('D').mat()
        cv_file.release()
        # FILE_STORAGE_READ
        # cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

        # # note we also have to specify the type to retrieve other wise we only get a
        # # FileNode object back instead of a matrix
        # self.camera_matrix = cv_file.getNode("K").mat()
        # self.dist_matrix = cv_file.getNode("D").mat()

        # cv_file.release()
        # return [self.camera_matrix, self.dist_matrix]



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
    
    #############################################################


    desired_aruco_dictionary = "DICT_ARUCO_ORIGINAL"
    
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
    
    if ARUCO_DICT.get(desired_aruco_dictionary, None) is None:
        print("[INFO] ArUCo tag of '{}' is not supported".format(args["type"]))
        sys.exit(0)

    
    # Load the ArUco dictionary
    print("[INFO] detecting '{}' markers...".format(
        desired_aruco_dictionary))
    this_aruco_dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT[desired_aruco_dictionary])
    this_aruco_parameters = cv2.aruco.DetectorParameters_create()
    
    cnt = 0

    while True:
        deltaTime = time.time() - start_time

        ######################## vision part ########################
        ret, depth_frame, color_frame, depth_frame_raw = dc.get_frame(detection = False)

     
        # Detect ArUco markers in the video frame
        (corners, ids, rejected) = cv2.aruco.detectMarkers(
        color_frame, this_aruco_dictionary, parameters=this_aruco_parameters)
        
        # Check that at least one ArUco marker was detected
        if len(corners) > 0:
            # Flatten the ArUco IDs list
            ids = ids.flatten()
            
            # Loop over the detected ArUco corners
            for (marker_corner, marker_id) in zip(corners, ids):
            
                # Extract the marker corners
                corners = marker_corner.reshape((4, 2))
                (top_left, top_right, bottom_right, bottom_left) = corners
                
                # Convert the (x,y) coordinate pairs to integers
                top_right = (int(top_right[0]), int(top_right[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
                top_left = (int(top_left[0]), int(top_left[1]))
                
                # Draw the bounding box of the ArUco detection
                cv2.line(color_frame, top_left, top_right, (0, 255, 0), 2)
                cv2.line(color_frame, top_right, bottom_right, (0, 255, 0), 2)
                cv2.line(color_frame, bottom_right, bottom_left, (0, 255, 0), 2)
                cv2.line(color_frame, bottom_left, top_left, (0, 255, 0), 2)
                
                # Calculate and draw the center of the ArUco marker
                center_x = int((top_left[0] + bottom_right[0]) / 2.0)
                center_y = int((top_left[1] + bottom_right[1]) / 2.0)
                cv2.circle(color_frame, (center_x, center_y), 4, (0, 0, 255), -1)
                
                # Draw the ArUco marker ID on the video frame
                # The ID is always located at the top_left of the ArUco marker
                cv2.putText(color_frame, str(marker_id), 
                (top_left[0], top_left[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)



                # aruco_marker_side_length = 0.045    
                # rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(
                #         corners,
                #         aruco_marker_side_length,
                #         dc.mtx,
                #         dc.dst)

                # # Print the pose for the ArUco marker
                # # The pose of the marker is with respect to the camera lens frame.
                # # Imagine you are looking through the camera viewfinder, 
                # # the camera lens frame's:
                # # x-axis points to the right
                # # y-axis points straight down towards your toes
                # # z-axis points straight ahead away from your eye, out of the camera
                # for i, marker_id in enumerate(marker_ids):
                
                #     # Store the translation (i.e. position) information
                #     transform_translation_x = tvecs[i][0][0]
                #     transform_translation_y = tvecs[i][0][1]
                #     transform_translation_z = tvecs[i][0][2]
            
                #     # Store the rotation information
                #     rotation_matrix = np.eye(4)
                #     rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
                #     r = R.from_matrix(rotation_matrix[0:3, 0:3])
                #     quat = r.as_quat()   
                    
                #     # Quaternion format     
                #     transform_rotation_x = quat[0] 
                #     transform_rotation_y = quat[1] 
                #     transform_rotation_z = quat[2] 
                #     transform_rotation_w = quat[3] 
                    
                #     # Euler angle format in radians
                #     roll_x, pitch_y, yaw_z = dc.euler_from_quaternion(transform_rotation_x, 
                #                                                 transform_rotation_y, 
                #                                                 transform_rotation_z, 
                #                                                 transform_rotation_w)
                    
                #     roll_x = math.degrees(roll_x)
                #     pitch_y = math.degrees(pitch_y)
                #     yaw_z = math.degrees(yaw_z)
                #     print("transform_translation_x: {}".format(transform_translation_x))
                #     print("transform_translation_y: {}".format(transform_translation_y))
                #     print("transform_translation_z: {}".format(transform_translation_z))
                #     print("roll_x: {}".format(roll_x))
                #     print("pitch_y: {}".format(pitch_y))
                #     print("yaw_z: {}".format(yaw_z))
                #     print()
                    
                #     # Draw the axes on the marker
                #     cv2.aruco.drawAxis(frame, dc.mtx, dc.dst, rvecs[i], tvecs[i], 0.05)
                

        point = dc.center_pos[0:2]
        #defining a point in the image
        cv2.circle(color_frame, point, 5, (0, 255, 0))
        #cv2.circle(depth_frame, point, 5, (0, 255, 0))

        #distance = depth_frame[point[1], point[0]]
        #distance = depth_frame[240,320]
        
        print("\nDistance = ", f"{depth_frame_raw.get_distance(point[0], point[1]):3.3f}")

        #depth and color frame of the image
        cv2.imshow("Depth Frame", depth_frame)
        cv2.imshow("Color Frame", color_frame)
        key = cv2.waitKey(1)

        if key == 27: # ESC key
            break
        elif key == 32:
            cv2.imwrite(f"imgcalib/image{cnt}.png",color_frame)
            cnt +=1

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

        # print('\nPhi = ', phi, ', Phi Ctrl = +/-', phiCtrl_M1_M2, ', Theta = ', theta, ', Theta Ctrl = +/-', thetaCtrl_M3_M4)
        # print('\nM1 (cm) = ', M1/10, ', M2 (cm) = ', M2/10, ', M3 (cm) = ', M3/10, ', M4 (cm) = ', M4/10)
        # print('\nTotal Length (cm) = ', totalLength/10)
        # print('\nEnd Tip Position (cm) = ', endTip_Pos)
        # print('\nTime = ', deltaTime)

robot.bus.shutdown()