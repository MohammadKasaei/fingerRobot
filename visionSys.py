#   https://automaticaddison.com/how-to-create-an-aruco-marker-using-opencv-python/

import cv2
import math
import numpy as np
import pybullet as p
from inspect import ArgSpec
import sys
from scipy.spatial.transform import Rotation as R
import time

class visionSys():    
 
    def __init__(self):        
        
        # Start the video stream
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3,640)
        self.cap.set(4,480)
        
        # Dictionary that was used to generate the ArUco marker
        self.aruco_dictionary_name = "DICT_ARUCO_ORIGINAL"
        
        # The different ArUco dictionaries built into the OpenCV library. 
        self.ARUCO_DICT = {
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
        
       
        # self.x_offset = -0.166
        # self.y_offset = +0.051
        # self.z_offset = -0.036
        
        self.x_offset = -0.0
        self.y_offset = -0.0
        self.z_offset = -0.0
        
        self.targetID = 1
        
        self.targetPose     = np.array([ 0.0, 0.0,  0.0, 0.0, 0.0,  0.0 ])
        self.targetPoseRaw  = np.array([ 0.0, 0.0,  0.0, 0.0, 0.0,  0.0 ])
        
        self.targetPoseFilterAlpha = 0.1
        
        self.cameraPos = np.array([-0.405,0,0])   
        self.cameraOrn = p.getQuaternionFromEuler(np.array([math.radians(-180),math.radians(-90),0]))
        
        self.PxMarker_EndTip = 0
        self.PyMarker_EndTip = 0
        self.PzMarker_EndTip = 0
        
        self.OxMarker_EndTip = 0
        self.OyMarker_EndTip = 0
        self.OzMarker_EndTip = 0

        # Side length of the ArUco marker in meters 
        self.aruco_marker_side_length = 0.018
        #self.aruco_marker_side_length = 0.012
        
        # Calibration parameters yaml file
        self.camera_calibration_parameters_filename = 'calibration_chessboard.yaml'
                
                # Check that we have a valid ArUco marker
        if self.ARUCO_DICT.get(self.aruco_dictionary_name, None) is None:
            print("[INFO] ArUCo tag of '{}' is not supported".format(
            ArgSpec["type"]))
            sys.exit(0)
        
        # Load the camera parameters from the saved file
        cv_file = cv2.FileStorage(self.camera_calibration_parameters_filename, cv2.FILE_STORAGE_READ) 
        self.mtx = cv_file.getNode('K').mat()
        self.dst = cv_file.getNode('D').mat()
        cv_file.release()
            
        # Load the ArUco dictionary
        print("[INFO] detecting '{}' markers...".format(
            self.aruco_dictionary_name))
        self.this_aruco_dictionary = cv2.aruco.Dictionary_get(self.ARUCO_DICT[self.aruco_dictionary_name])
        self.this_aruco_parameters = cv2.aruco.DetectorParameters_create()
        
        # Start the video stream
        cap = cv2.VideoCapture(0)
        ret, self.frame = self.cap.read()
        self.lastFrameUpdate = time.time()
        self.drawAxis = False
        
        self.markerPose = np.zeros((10,6))

    def euler_from_quaternion(self, x, y, z, w):
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
    
    def getMarkerPose(self,markerID):
        
        # ee_pos = (1-lpf_alpha)*ee_pos + lpf_alpha*tvecs[i][0][0:3] 
        # cv2.circle(frame, (ee_pos[0],ee_pos[1]), 10, (200,200,0))
        # Store the translation (i.e. position) information 
        self.transform_translation_x_MkID = self.tvecs[markerID][0][0] 
        self.transform_translation_y_MkID = self.tvecs[markerID][0][1] 
        self.transform_translation_z_MkID = self.tvecs[markerID][0][2] 
        # Store the rotation information
        rotation_matrix_MkID = np.eye(4)
        rotation_matrix_MkID[0:3, 0:3] = cv2.Rodrigues(np.array(self.rvecs[markerID][0]))[0]
        r_MkID = R.from_matrix(rotation_matrix_MkID[0:3, 0:3])
        quat_MkID = r_MkID.as_quat()   

        # Quaternion format     
        transform_rotation_x_MkID = quat_MkID[0] 
        transform_rotation_y_MkID = quat_MkID[1] 
        transform_rotation_z_MkID = quat_MkID[2] 
        transform_rotation_w_MkID = quat_MkID[3] 

        # Euler angle format in radians
        roll_x_MkID, pitch_y_MkID, yaw_z_MkID = self.euler_from_quaternion(transform_rotation_x_MkID, 
                                            transform_rotation_y_MkID, 
                                            transform_rotation_z_MkID, 
                                            transform_rotation_w_MkID)

        objPos_MkID = np.array([self.transform_translation_x_MkID,self.transform_translation_y_MkID,self.transform_translation_z_MkID]) 
        objOrn_MkID = p.getQuaternionFromEuler(np.array([roll_x_MkID, pitch_y_MkID, yaw_z_MkID]))  
        
        obj2robot_MkID = p.multiplyTransforms(self.cameraPos,self.cameraOrn,objPos_MkID,objOrn_MkID)
        objEulerOrn_MkID = p.getEulerFromQuaternion(obj2robot_MkID[1])

        #print(f"Marker Orientation (End-Tip): {math.degrees(objEulerOrn[0]):3.3f}\t{math.degrees(objEulerOrn[1]):3.3f}\t{math.degrees(objEulerOrn[2]):3.3f}")
    
        self.PxMarker_EndTip = obj2robot_MkID[0][0] + self.x_offset
        self.PyMarker_EndTip = obj2robot_MkID[0][1] + self.y_offset
        self.PzMarker_EndTip = obj2robot_MkID[0][2] + self.z_offset
        
        # self.PxMarker_EndTip = ((1-self.targetPoseFilterAlpha)*self.PxMarker_EndTip) + (self.targetPoseFilterAlpha*self.markerPose[markerID][0])
        # self.PyMarker_EndTip = ((1-self.targetPoseFilterAlpha)*self.PyMarker_EndTip) + (self.targetPoseFilterAlpha*self.markerPose[markerID][1])
        # self.PzMarker_EndTip = ((1-self.targetPoseFilterAlpha)*self.PzMarker_EndTip) + (self.targetPoseFilterAlpha*self.markerPose[markerID][2])
        
        self.OxMarker_EndTip = obj2robot_MkID[1][0]
        self.OyMarker_EndTip = obj2robot_MkID[1][1]
        self.OzMarker_EndTip = obj2robot_MkID[1][2] 
        
        return np.array((self.PxMarker_EndTip, self.PyMarker_EndTip, self.PzMarker_EndTip, self.OxMarker_EndTip, self.OyMarker_EndTip, self.OzMarker_EndTip))
        
    def detectMarkers(self):
        
        # Detect ArUco markers in the video frame
        (corners, marker_ids, rejected) = cv2.aruco.detectMarkers(
        self.frame, self.this_aruco_dictionary, parameters=self.this_aruco_parameters,
        cameraMatrix=self.mtx, distCoeff=self.dst)
        
        # Check that at least one ArUco marker was detected
        if marker_ids is not None:
        
            # Draw a square around detected markers in the video frame
            cv2.aruco.drawDetectedMarkers(self.frame, corners, marker_ids)
            
            # Get the rotation and translation vectors
            self.rvecs, self.tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                self.aruco_marker_side_length,
                self.mtx,
                self.dst)

            for i, marker_id in enumerate(marker_ids):
                if (marker_id<10):
                    self.markerPose[marker_id] = self.getMarkerPose(i)
                    # self.markerPose[marker_id] = ((1-self.targetPoseFilterAlpha)*self.markerPose[marker_id]) + (self.targetPoseFilterAlpha*self.getMarkerPose(i))

                    
                    if self.drawAxis:
                        cv2.aruco.drawAxis(self.frame, self.mtx, self.dst, self.rvecs[i], self.tvecs[i], 0.025) 
                    
                    if marker_id == self.targetID:                
                        self.targetPose    = ((1-self.targetPoseFilterAlpha)*self.targetPose) + (self.targetPoseFilterAlpha*self.markerPose[marker_id])
                        self.targetPoseRaw = self.markerPose[marker_id]
                        

    def visualize(self):
        
        if self.drawAxis:
            # Draw the axes on the robot base
            cv2.line(self.frame, (320,0),(320,480), (150,150,150),2)
            cv2.line(self.frame, (0,240),(640,240), (150,150,150),2)
            cv2.line(self.frame, (320,240),(345,215), (0,0,200),3)
            cv2.line(self.frame, (320,190),(320,240), (0,200,0),3)
            cv2.line(self.frame, (320,240),(370,240), (200,0,0),3)
            
        # Display the resulting frame
        cv2.imshow('Finger Robot',self.frame)
        
    def update(self,freq,vis,waitForKey=1):
        
        if  time.time() - self.lastFrameUpdate > 1.0/freq :
            self.lastFrameUpdate = time.time()
            ret, self.frame = self.cap.read()
        
        self.detectMarkers()

        key = cv2.waitKey(waitForKey)
        
        if key == ord('v') or key == ord('V') :
           self.drawAxis  = False if self.drawAxis  else True 
        
        if (vis): 
            self.visualize()
            
        return key

         
        
