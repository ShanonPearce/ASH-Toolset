# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 21:54:23 2024

There are three major steps:
1. Detect and crop the human faces in the video frame.
2. Run facial landmark detection on the face image.
3. Estimate the pose by solving a PnP problem.

This code is derived from the following project:
https://github.com/yinguobing/head-pose-estimation
"""


import cv2
import numpy as np



from ash_toolset import face_detection
from ash_toolset import mark_detection
from ash_toolset import pose_estimation
from ash_toolset import utils
from ash_toolset import constants as CN

import logging
from os.path import join as pjoin

logger = logging.getLogger(__name__)
log_info=1

#print(__doc__)
#print("OpenCV version: {}".format(cv2.__version__))


def setup_detection(camera_idx=0,gui_logger=None):
    """
    function to perform setup of detectors
    returns detector objects
    """
    
    detector_arr = []
    
    try:
        in_file_face = pjoin(CN.DATA_DIR_ASSETS, 'face_detector.onnx')
        in_file_mark = pjoin(CN.DATA_DIR_ASSETS, 'face_landmarks.onnx')
        
        # Before estimation started, there are some startup works to do.
    
        # Initialize the video source from webcam or video file.
        video_src = camera_idx
        apiPreference=cv2.CAP_MSMF
        cap = cv2.VideoCapture(video_src, apiPreference=apiPreference)


        # Get the frame size. This will be used by the following detectors.
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


        # Setup a face detector to detect human faces.
        face_detector = face_detection.FaceDetector(in_file_face)

        # Setup a mark detector to detect landmarks.
        mark_detector = mark_detection.MarkDetector(in_file_mark)
    
        # Setup a pose estimator to solve pose.
        pose_estimator = pose_estimation.PoseEstimator(frame_width, frame_height)
    
        detector_arr = [face_detector,mark_detector,pose_estimator]
        
        
        # # Read a frame.
        # frame_got, frame = cap.read()
        # if frame_got is False or frame_got is None:
        #     print("no frame detected")
        #     return detector_arr #return value?
        
        cap.release()
        cv2.destroyAllWindows()
        
        return detector_arr
        
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        log_string = 'Failed to setup detectors'
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string)
            
        return detector_arr
  
def setup_video_cap(camera_idx=0, frames_to_read=3, gui_logger=None):
    """
    function to perform setup of cv2 video capture object
    returns cv2 video capture object or None
    """
    try:
        # Initialize the video source from webcam or video file.
        video_src = camera_idx
        cap = cv2.VideoCapture(video_src)
        
        # Read a frame.
        for x in range(frames_to_read):
            frame_got, frame = cap.read()
        if frame_got is False or frame_got is None:
            raise ValueError('no frame detected')
            
        return cap
    
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        log_string = 'Failed to setup cv2 video capture object'
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string)
    



def estimate_head_pose(detector_arr=[], cap=None, camera_idx=0, frames_to_read=1, gui_logger=None):
    """
    function to estimate head pose based on webcam input
    returns estimated elevation angle and azimuth angle of subject's head relative to webcam
    """
    
    pose_arr = []
    
    try:

        # Before estimation started, there are some startup works to do.
        video_src = camera_idx
        if cap == None:
            # Initialize the video source from webcam or video file.
            cap = cv2.VideoCapture(video_src)
            frames_to_read=3#need to read a few frames to detect a face
            #print(f"Video source: {video_src}")
        
        
    
        # Get the frame size. This will be used by the following detectors.
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #print(str(frame_width))
        
        #use detectors from provided array if not empty
        if detector_arr:
            # Setup a face detector to detect human faces.
            face_detector = detector_arr[0]
            # Setup a mark detector to detect landmarks.
            mark_detector = detector_arr[1]
            # Setup a pose estimator to solve pose.
            pose_estimator = detector_arr[2]
        else:
            in_file_face = pjoin(CN.DATA_DIR_ASSETS, 'face_detector.onnx')
            in_file_mark = pjoin(CN.DATA_DIR_ASSETS, 'face_landmarks.onnx')
            
            # Setup a face detector to detect human faces.
            face_detector = face_detection.FaceDetector(in_file_face)
            # Setup a mark detector to detect landmarks.
            mark_detector = mark_detection.MarkDetector(in_file_mark)
            # Setup a pose estimator to solve pose.
            pose_estimator = pose_estimation.PoseEstimator(frame_width, frame_height)


        # Read a frame.
        for x in range(frames_to_read):
            frame_got, frame = cap.read()
        if frame_got is False or frame_got is None:
            raise ValueError('no frame detected')

        

        # If the frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Step 1: Get faces from current frame.
        faces, _ = face_detector.detect(frame, 0.7)
        

        # Any valid face found?
        if len(faces) > 0:
            
            #print('step 2 started')
 
            # Step 2: Detect landmarks. Crop and feed the face area into the
            # mark detector. Note only the first face will be used for
            # demonstration.
            face = utils.refine(faces, frame_width, frame_height, 0.15)[0]
            x1, y1, x2, y2 = face[:4].astype(int)
            patch = frame[y1:y2, x1:x2]

            # Run the mark detection.
            marks = mark_detector.detect([patch])[0].reshape([68, 2])

            # Convert the locations from local face area to the global image.
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            # Step 3: Try pose estimation with 68 points.
            pose = pose_estimator.solve(marks)
            
            
            # Solve the pitch, yaw and roll angels.
            r_mat, _ = cv2.Rodrigues(pose[0])
            p_mat = np.hstack((r_mat, np.array([[0], [0], [0]])))
            _, _, _, _, _, _, u_angle = cv2.decomposeProjectionMatrix(p_mat)
            pitch, yaw, roll = u_angle.flatten()
            pose_arr = [pitch, yaw, roll]
            
            
    
        else:
            raise ValueError('no faces detected')
    
    
        #cap.release()
        #cv2.destroyAllWindows()
        
        return pose_arr
    
    except Exception as ex:
        # logging.error("Error occurred", exc_info = ex)
        # log_string = 'Failed to perform head pose estimation'
            
        return pose_arr

def cleanup_capture(cap=None, gui_logger=None):
    """
    function to perform setup of cv2 video capture object
    returns cv2 video capture object or None
    """
    try:
        cap.release()
        cv2.destroyAllWindows()
    
    except Exception as ex:
        logging.error("Error occurred", exc_info = ex)
        log_string = 'Failed to cleanup cv2 video capture object'
        if CN.LOG_GUI == 1 and gui_logger != None:
            gui_logger.log_info(log_string)
    
