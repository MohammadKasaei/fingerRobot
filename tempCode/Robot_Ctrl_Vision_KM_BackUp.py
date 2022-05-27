
# https://automaticaddison.com/how-to-create-an-aruco-marker-using-opencv-python/

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

from FingerRobot import FingerRobot
from visionSys import visionSys



#Global Variables
global lengthOffset, allMotorVal
global angleTune
global M1, M2, M3, M4
global thetaCtrl_M3_M4, theta
global phiCtrl_M1_M2, phi
global endTip_PosX, endTip_PosY, endTip_PosZ
global endTip_Pos
global deltaTime
global lengthCtrlKey
global inputMode

global lengthCtrlMarker_New, phiMarker_New, thetaMarker_New
global PzMarker_New, PyMarker_New, PxMarker_New
global PzMarker_EndTip, PyMarker_EndTip, PxMarker_EndTip 

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
aruco_marker_side_length = 0.024
 
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
 
 
def main():
  """
  Main method of the program.
  """
  # Check that we have a valid ArUco marker
  if ARUCO_DICT.get(aruco_dictionary_name, None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported".format(
      ArgSpec["type"]))
    sys.exit(0)
 
  # Load the camera parameters from the saved file
  cv_file = cv2.FileStorage(camera_calibration_parameters_filename, cv2.FILE_STORAGE_READ) 
  mtx = cv_file.getNode('K').mat()
  dst = cv_file.getNode('D').mat()
  cv_file.release()
     
  # Load the ArUco dictionary
  print("[INFO] detecting '{}' markers...".format(
    aruco_dictionary_name))
  this_aruco_dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_dictionary_name])
  this_aruco_parameters = cv2.aruco.DetectorParameters_create()
   
  # # Start the video stream
  # cap = cv2.VideoCapture(0)

  cnt = 0 

  lengthOffset = 77   # length offset
  angleTune = 6       # angle tuning coef (experimentally!)

  lengthCtrlKey = 0
  theta = 0
  phi = 0

  lengthCtrlMarker_New = 0
  thetaMarker_New = 0
  phiMarker_New = 0

  # ModeCtrl
  # Mode Ctrl = 0 --> Keyboard Input
  # Mode Ctrl = 1 --> Object Tracking
  inputMode = 0

  vis = False

  x_offset = -0.036
  y_offset = +0.036
  z_offset = -0.18

  lpf_alpha = 0.2
  ee_pos = np.zeros(3)
  ee_orn = np.zeros(3)

  # robotPos = np.array([0,0,0]) 
  # robotOrn = p.getQuaternionFromEuler(np.array([0,0,0]))

  cameraPos = np.array([-0.50,0,0])   
  cameraOrn = p.getQuaternionFromEuler(np.array([math.radians(-180),math.radians(-90),0]))

  #cameraPos = np.array([0,0,0])   
  #cameraOrn = p.getQuaternionFromEuler(np.array([math.radians(0),math.radians(0),0]))

  #objPos = np.array([0.09,-0.024,0.50]) 
  #objOrn = p.getQuaternionFromEuler(np.array([0,0,0]))  
  
  # obj2robot = p.multiplyTransforms(cameraPos,cameraOrn,objPos,objOrn)
  # objEulerOrn = p.getEulerFromQuaternion(obj2robot[1])
  # print(f"position: {obj2robot[0]} r {math.degrees(objEulerOrn[0])} p {math.degrees(objEulerOrn[1])} y {math.degrees(objEulerOrn[2])}" )
  
  #vision = visionSys()
  #while (True):
  #      vision.update(vis=True)
      

  while(True):
  
    # Capture frame-by-frame
    # This method returns True/False as well
    # as the video frame.
    ret, frame = cap.read()  
     
    # Detect ArUco markers in the video frame
    (corners, marker_ids, rejected) = cv2.aruco.detectMarkers(
      frame, this_aruco_dictionary, parameters=this_aruco_parameters,
      cameraMatrix=mtx, distCoeff=dst)
       
    # Check that at least one ArUco marker was detected
    if marker_ids is not None:
 
      # Draw a square around detected markers in the video frame
      cv2.aruco.drawDetectedMarkers(frame, corners, marker_ids)
       
      # Get the rotation and translation vectors
      rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(
        corners,
        aruco_marker_side_length,
        mtx,
        dst)
         
      # Print the pose for the ArUco marker
      # The pose of the marker is with respect to the camera lens frame.
      # Imagine you are looking through the camera viewfinder, 

      # the camera lens frame's:

      # X-axis points to the right
      # Y-axis points straight down towards your toes
      # Z-axis points straight ahead away from your eye, out of the camera

      for i, marker_id in enumerate(marker_ids):
       
        if (marker_id == 1):
              # ee_pos = (1-lpf_alpha)*ee_pos + lpf_alpha*tvecs[i][0][0:3] 
          # cv2.circle(frame, (ee_pos[0],ee_pos[1]), 10, (200,200,0))
          # Store the translation (i.e. position) information 
          transform_translation_x_MkID1 = tvecs[i][0][0] + x_offset
          transform_translation_y_MkID1 = tvecs[i][0][1] + y_offset
          transform_translation_z_MkID1 = tvecs[i][0][2] + z_offset
          # Store the rotation information
          rotation_matrix_MkID1 = np.eye(4)
          rotation_matrix_MkID1[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
          r_MkID1 = R.from_matrix(rotation_matrix_MkID1[0:3, 0:3])
          quat_MkID1 = r_MkID1.as_quat()   
 
          # Quaternion format     
          transform_rotation_x_MkID1 = quat_MkID1[0] 
          transform_rotation_y_MkID1 = quat_MkID1[1] 
          transform_rotation_z_MkID1 = quat_MkID1[2] 
          transform_rotation_w_MkID1 = quat_MkID1[3] 
 
          # Euler angle format in radians
          roll_x_MkID1, pitch_y_MkID1, yaw_z_MkID1 = euler_from_quaternion(transform_rotation_x_MkID1, 
                                               transform_rotation_y_MkID1, 
                                               transform_rotation_z_MkID1, 
                                               transform_rotation_w_MkID1)
 
          objPos_MkID1 = np.array([transform_translation_x_MkID1,transform_translation_y_MkID1,transform_translation_z_MkID1]) 
          objOrn_MkID1 = p.getQuaternionFromEuler(np.array([roll_x_MkID1, pitch_y_MkID1, yaw_z_MkID1]))  
        
          obj2robot_MkID1 = p.multiplyTransforms(cameraPos,cameraOrn,objPos_MkID1,objOrn_MkID1)
          objEulerOrn_MkID1 = p.getEulerFromQuaternion(obj2robot_MkID1[1])

          #print(f"Marker Orientation (End-Tip): {math.degrees(objEulerOrn[0]):3.3f}\t{math.degrees(objEulerOrn[1]):3.3f}\t{math.degrees(objEulerOrn[2]):3.3f}")
    
          PxMarker_EndTip = obj2robot_MkID1[0][0]
          PyMarker_EndTip = obj2robot_MkID1[0][1]
          PzMarker_EndTip = obj2robot_MkID1[0][2]
          
          if (vis): 
            # Draw the axes on the marker
            cv2.aruco.drawAxis(frame, mtx, dst, rvecs[i], tvecs[i], 0.025)
       
        if (marker_id == 2):
              # ee_pos = (1-lpf_alpha)*ee_pos + lpf_alpha*tvecs[i][0][0:3] 
          # cv2.circle(frame, (ee_pos[0],ee_pos[1]), 10, (200,200,0))
          # Store the translation (i.e. position) information 
          transform_translation_x_MkID2 = tvecs[i][0][0] + x_offset
          transform_translation_y_MkID2 = tvecs[i][0][1] + y_offset
          transform_translation_z_MkID2 = tvecs[i][0][2] + z_offset
          # Store the rotation information
          rotation_matrix_MkID2 = np.eye(4)
          rotation_matrix_MkID2[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
          r_MkID2 = R.from_matrix(rotation_matrix_MkID2[0:3, 0:3])
          quat_MkID2 = r_MkID2.as_quat()   
 
          # Quaternion format     
          transform_rotation_x_MkID2 = quat_MkID2[0] 
          transform_rotation_y_MkID2 = quat_MkID2[1] 
          transform_rotation_z_MkID2 = quat_MkID2[2] 
          transform_rotation_w_MkID2 = quat_MkID2[3] 
 
          # Euler angle format in radians
          roll_x_MkID2, pitch_y_MkID2, yaw_z_MkID2 = euler_from_quaternion(transform_rotation_x_MkID2, 
                                               transform_rotation_y_MkID2, 
                                               transform_rotation_z_MkID2, 
                                               transform_rotation_w_MkID2)
 
          objPos_MkID2 = np.array([transform_translation_x_MkID2,transform_translation_y_MkID2,transform_translation_z_MkID2]) 
          objOrn_MkID2 = p.getQuaternionFromEuler(np.array([roll_x_MkID2, pitch_y_MkID2, yaw_z_MkID2]))  
        
          obj2robot_MkID2 = p.multiplyTransforms(cameraPos,cameraOrn,objPos_MkID2,objOrn_MkID2)
          objEulerOrn_MkID2 = p.getEulerFromQuaternion(obj2robot_MkID2[1])

          PxMarker_EndTip = obj2robot_MkID2[0][0]
          PyMarker_EndTip = obj2robot_MkID2[0][1]
          PzMarker_EndTip = obj2robot_MkID2[0][2]
          
          if (vis): 
            # Draw the axes on the marker
            cv2.aruco.drawAxis(frame, mtx, dst, rvecs[i], tvecs[i], 0.025)

        if (marker_id == 3):
              # ee_pos = (1-lpf_alpha)*ee_pos + lpf_alpha*tvecs[i][0][0:3] 
          # cv2.circle(frame, (ee_pos[0],ee_pos[1]), 10, (200,200,0))
          # Store the translation (i.e. position) information 
          transform_translation_x_MkID3 = tvecs[i][0][0] + x_offset
          transform_translation_y_MkID3 = tvecs[i][0][1] + y_offset
          transform_translation_z_MkID3 = tvecs[i][0][2] + z_offset
          # Store the rotation information
          rotation_matrix_MkID3 = np.eye(4)
          rotation_matrix_MkID3[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
          r_MkID3 = R.from_matrix(rotation_matrix_MkID3[0:3, 0:3])
          quat_MkID3 = r_MkID3.as_quat()   
 
          # Quaternion format     
          transform_rotation_x_MkID3 = quat_MkID3[0] 
          transform_rotation_y_MkID3 = quat_MkID3[1] 
          transform_rotation_z_MkID3 = quat_MkID3[2] 
          transform_rotation_w_MkID3 = quat_MkID3[3] 
 
          # Euler angle format in radians
          roll_x_MkID3, pitch_y_MkID3, yaw_z_MkID3 = euler_from_quaternion(transform_rotation_x_MkID3, 
                                               transform_rotation_y_MkID3, 
                                               transform_rotation_z_MkID3, 
                                               transform_rotation_w_MkID3)
 
          objPos_MkID3 = np.array([transform_translation_x_MkID3,transform_translation_y_MkID3,transform_translation_z_MkID3]) 
          objOrn_MkID3 = p.getQuaternionFromEuler(np.array([roll_x_MkID3, pitch_y_MkID3, yaw_z_MkID3]))  
        
          obj2robot_MkID3 = p.multiplyTransforms(cameraPos,cameraOrn,objPos_MkID3,objOrn_MkID3)
          objEulerOrn_MkID3 = p.getEulerFromQuaternion(obj2robot_MkID3[1])

          PxMarker_EndTip = obj2robot_MkID3[0][0]
          PyMarker_EndTip = obj2robot_MkID3[0][1]
          PzMarker_EndTip = obj2robot_MkID3[0][2]
          
          if (vis): 
            # Draw the axes on the marker
            cv2.aruco.drawAxis(frame, mtx, dst, rvecs[i], tvecs[i], 0.025)

        if (marker_id == 4):
          # ee_pos = (1-lpf_alpha)*ee_pos + lpf_alpha*tvecs[i][0][0:3] 
          # cv2.circle(frame, (ee_pos[0],ee_pos[1]), 10, (200,200,0))
          # Store the translation (i.e. position) information 
          transform_translation_x_MkID4 = tvecs[i][0][0] + x_offset
          transform_translation_y_MkID4 = tvecs[i][0][1] + y_offset
          transform_translation_z_MkID4 = tvecs[i][0][2] + z_offset
          # Store the rotation information
          rotation_matrix_MkID4 = np.eye(4)
          rotation_matrix_MkID4[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
          r_MkID4 = R.from_matrix(rotation_matrix_MkID4[0:3, 0:3])
          quat_MkID4 = r_MkID4.as_quat()   
 
          # Quaternion format     
          transform_rotation_x_MkID4 = quat_MkID4[0] 
          transform_rotation_y_MkID4 = quat_MkID4[1] 
          transform_rotation_z_MkID4 = quat_MkID4[2] 
          transform_rotation_w_MkID4 = quat_MkID4[3] 
 
          # Euler angle format in radians
          roll_x_MkID4, pitch_y_MkID4, yaw_z_MkID4 = euler_from_quaternion(transform_rotation_x_MkID4, 
                                               transform_rotation_y_MkID4, 
                                               transform_rotation_z_MkID4, 
                                               transform_rotation_w_MkID4)
 
          objPos_MkID4 = np.array([transform_translation_x_MkID4,transform_translation_y_MkID4,transform_translation_z_MkID4]) 
          objOrn_MkID4 = p.getQuaternionFromEuler(np.array([roll_x_MkID4, pitch_y_MkID4, yaw_z_MkID4]))  
        
          obj2robot_MkID4 = p.multiplyTransforms(cameraPos,cameraOrn,objPos_MkID4,objOrn_MkID4)
          objEulerOrn_MkID4 = p.getEulerFromQuaternion(obj2robot_MkID4[1])

          if (vis): 
            # Draw the axes on the marker
            cv2.aruco.drawAxis(frame, mtx, dst, rvecs[i], tvecs[i], 0.025)      

        if (marker_id == 5):
              # ee_pos = (1-lpf_alpha)*ee_pos + lpf_alpha*tvecs[i][0][0:3] 
          # cv2.circle(frame, (ee_pos[0],ee_pos[1]), 10, (200,200,0))
          # Store the translation (i.e. position) information 
          transform_translation_x_MkID5 = tvecs[i][0][0] + x_offset
          transform_translation_y_MkID5 = tvecs[i][0][1] + y_offset
          transform_translation_z_MkID5 = tvecs[i][0][2] + z_offset
          # Store the rotation information
          rotation_matrix_MkID5 = np.eye(4)
          rotation_matrix_MkID5[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
          r_MkID5 = R.from_matrix(rotation_matrix_MkID5[0:3, 0:3])
          quat_MkID5 = r_MkID5.as_quat()   
 
          # Quaternion format     
          transform_rotation_x_MkID5 = quat_MkID5[0] 
          transform_rotation_y_MkID5 = quat_MkID5[1] 
          transform_rotation_z_MkID5 = quat_MkID5[2] 
          transform_rotation_w_MkID5 = quat_MkID5[3] 
 
          # Euler angle format in radians
          roll_x_MkID5, pitch_y_MkID5, yaw_z_MkID5 = euler_from_quaternion(transform_rotation_x_MkID5, 
                                               transform_rotation_y_MkID5, 
                                               transform_rotation_z_MkID5, 
                                               transform_rotation_w_MkID5)
 
          objPos_MkID5 = np.array([transform_translation_x_MkID5,transform_translation_y_MkID5,transform_translation_z_MkID5]) 
          objOrn_MkID5 = p.getQuaternionFromEuler(np.array([roll_x_MkID5, pitch_y_MkID5, yaw_z_MkID5]))  
        
          obj2robot_MkID5 = p.multiplyTransforms(cameraPos,cameraOrn,objPos_MkID5,objOrn_MkID5)
          objEulerOrn_MkID5 = p.getEulerFromQuaternion(obj2robot_MkID5[1])

          PxMarker_EndTip = obj2robot_MkID5[0][0]
          PyMarker_EndTip = obj2robot_MkID5[0][1]
          PzMarker_EndTip = obj2robot_MkID5[0][2]
          
          if (vis): 
            # Draw the axes on the marker
            cv2.aruco.drawAxis(frame, mtx, dst, rvecs[i], tvecs[i], 0.025)

    if (vis): 
        # Draw the axes on the robot base
        cv2.line(frame, (320,0),(320,480), (150,150,150),2)
        cv2.line(frame, (0,240),(640,240), (150,150,150),2)
        cv2.line(frame, (320,240),(345,215), (0,0,200),3)
        cv2.line(frame, (320,190),(320,240), (0,200,0),3)
        cv2.line(frame, (320,240),(370,240), (200,0,0),3)

    # Display the resulting frame
    cv2.imshow('Finger Robot',frame)
          
    # If "Esc" is pressed on the keyboard, 
    # exit this loop
    key = cv2.waitKey(1)
    if key == 113: # 'q' key
      break
    
    # elif key == 32: # Space key
    #   cv2.imwrite(f"imgcalib/image{cnt}.png",frame)
    #   cnt +=1
   
    deltaTime = time.time() - start_time

    ######################## control part ########################
    if key == 27:
      lengthCtrlKey = 0
      theta = 0
      phi = 0
      
    elif key == 43:
        lengthCtrlKey += 1
        if lengthCtrlKey >= 75:
            lengthCtrlKey = 75
    
    elif key == 45:
        lengthCtrlKey -=1
        if lengthCtrlKey <= 0:
            lengthCtrlKey = 0

    elif key == 56:
        theta += 1
        if theta >= 90:
          theta = 90

    elif key == 50:
        theta -= 1
        if theta <= - 90:
          theta = -90

    elif key == 54:
        phi += 1
        if phi >= 90:
          phi = 90

    elif key == 52:
        phi -=1
        if phi <= -90:
          phi = -90

    elif key == ord('v') or key == ord('V') :
       vis = False if vis else True 

    # Marker ID4 detected
    # PzMarker_New = (((obj2robot_MkID4[0][2]))*(math.sin((thetaMarker_New+90) * (math.pi/180)) + math.sin((phiMarker_New+90) * (math.pi/180))))/2
    PzMarker_New = (((obj2robot_MkID4[0][2]))*(math.sin(math.radians(thetaMarker_New+90)) + math.sin(math.radians(phiMarker_New+90))))/2

    lengthCtrlMarker_New = int(PzMarker_New*1000 - lengthOffset)

    PyMarker_New = obj2robot_MkID4[0][1] + 0.036
    tmp_y = lengthOffset + lengthCtrlMarker_New

    if tmp_y == 0:
          tmp_y = 1

    tmp_y_thetaCtrl = PyMarker_New*1000/(tmp_y)

    if tmp_y_thetaCtrl >= 1:
          tmp_y_thetaCtrl = 1
    elif tmp_y_thetaCtrl <= -1:
      tmp_y_thetaCtrl = -1

    thetaMarker_New = -(((math.acos(tmp_y_thetaCtrl)) * 180/math.pi) - 90)

    PxMarker_New = obj2robot_MkID4[0][0]
    tmp_x = lengthOffset + lengthCtrlMarker_New

    if tmp_x == 0:
        tmp_x = 1

    tmp_x_phiCtrl = PxMarker_New*1000/(tmp_x)

    if tmp_x_phiCtrl >= 1:
              tmp_x_phiCtrl = 1
    elif tmp_x_phiCtrl <= -1:
      tmp_x_phiCtrl = -1

    phiMarker_New = -(((math.acos(tmp_x_phiCtrl)) * 180/math.pi) - 90)
    
    if lengthCtrlMarker_New <= 1:
        lengthCtrlMarker_New = 1

    if lengthCtrlMarker_New >= 75:
        lengthCtrlMarker_New = 75

    # Robot  
    if inputMode == 0:
      allMotorVal = lengthCtrlKey  #+ lengthCtrlMarker_New  # length control addition/subtraction from length offset
      phiCtrl_M1_M2 = phi/angleTune       # theta control (Z-Y plane)
      thetaCtrl_M3_M4 = theta/angleTune   # phi control (Z-X plane)
    elif inputMode == 1:
      allMotorVal =  lengthCtrlMarker_New  # length control addition/subtraction from length offset
      phiCtrl_M1_M2 = phiMarker_New/angleTune       # theta control (Z-Y plane)
      thetaCtrl_M3_M4 = thetaMarker_New/angleTune   # phi control (Z-X plane)

    M1 = allMotorVal+phiCtrl_M1_M2      # phi ==> rotation around y-axis (Z-X plane)
    M2 = allMotorVal-phiCtrl_M1_M2      # phi ==> rotation around y-axis (Z-X plane)
    M3 = allMotorVal+thetaCtrl_M3_M4    # theta ==> rotation around x-axis (Z-Y plane)
    M4 = allMotorVal-thetaCtrl_M3_M4    # theta ==> rotation around x-axis (Z-Y plane)

    totalLength = lengthOffset + allMotorVal # total length of the robot

    # End-Tip Position Calculation
    endTip_PosX = (totalLength * math.cos((phi-90) * (math.pi/180)))
    endTip_PosY = (totalLength * math.cos((theta-90) * (math.pi/180))) 
    endTip_PosZ = ((totalLength/2) * (math.sin((theta+90) * (math.pi/180)) + math.sin((phi+90) * (math.pi/180))))
    endTip_Pos = [endTip_PosX/1000, endTip_PosY/1000, endTip_PosZ/1000]

    robot.sendMessage ([M1,M2,M3,M4])

    if inputMode == 0:
      print("\nRobot Length Ctrl = ", lengthCtrlKey)
      print("Theta Value Ctrl (Z-Y plane) = ", theta)
      print("Phi Value Ctrl (Z-X plane) = ", phi)
      print('End Tip Position (m) = ', endTip_Pos)
      print('Time = ', f"{deltaTime:3.3f}") 

    if inputMode == 1:
      print("Px_new_marker = ", PxMarker_New, "\tPy_new_marker = ", PyMarker_New, "\tPz_new_marker = ", PzMarker_New)
      print("lengthCtrlMarker_New (mm) = ", lengthCtrlMarker_New)
      print("thetaMarker_New (deg)", thetaMarker_New)
      print("phiMarker_New (deg)", phiMarker_New)
      print('End Tip Position (m) = ', endTip_Pos)
      print('Time = ', f"{deltaTime:3.3f}") 
        
    #print('Phi = ', phi, ', Phi Ctrl = +/-', phiCtrl_M1_M2, ', Theta = ', theta, ', Theta Ctrl = +/-', thetaCtrl_M3_M4)
    #print('M1 (cm) = ', M1/10, ', M2 (cm) = ', M2/10, ', M3 (cm) = ', M3/10, ', M4 (cm) = ', M4/10)
    #print('Total Length (m) = ', totalLength/1000)
    
    #print(f"Marker Position (Target): {obj2robot_MkID4[0][0]:3.3f}\t{obj2robot_MkID4[0][1]:3.3f}\t{obj2robot_MkID4[0][2]:3.3f}")
    #print(f"Marker Orientation (Target): {math.degrees(objEulerOrn_MkID4[0]):3.3f}\t{math.degrees(objEulerOrn_MkID4[1]):3.3f}\t{math.degrees(objEulerOrn_MkID4[2]):3.3f}")
    
  # Close down the video stream
  cap.release()
  cv2.destroyAllWindows()
   
if __name__ == '__main__':
  start_time = time.time()    
  robot = FingerRobot()
  print(__doc__)
  main()