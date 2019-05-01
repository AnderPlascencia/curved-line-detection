#libreria de funciones
from functions.utils import *

import numpy as np
import cv2

#libreria ros
import rospy
from std_msgs.msg import String

#convertidor entre ros y opencv
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image

# import pandas as pd
# import os
# import glob
# from matplotlib.pyplot import subplots as plt
# from matplotlib.pyplot import subplots_adjust as plta
# import pickle

red = (0, 0, 255)
blue = (255, 0, 0)
green = (0, 255, 0)
kernel = np.ones((3,3),np.uint8)
camara = "project_video.mp4"

cap = cv2.VideoCapture(camara)

#lectura de mensaje
def image_callback(msg):
    cv2_img = CvBridge().imgmsg_to_cv2(msg, "bgr8")

image_topic = "/videofile/image_raw"

rospy.init_node('image_publish', anonymous=True)
filtered_image = rospy.Publisher('filtered_image', Image, queue_size=1)
rospy.Subscriber(image_topic, Image, image_callback)

while(True):

    # lectura de camara
    ret, frame = cap.read()
    
    if not ret:
        print( "Error en la lectura de imagen" )
        cap = cv2.VideoCapture(camara)
    
    # width, height, channel = frame.shape
    width = 640
    height = 480
    frame = cv2.resize(frame, (width, height))
    midpoint = int(frame.shape[1]/2)
    # img = cv2.imread('Curved-Lane-Lines/test_images/test3.jpg')
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dst = pipeline(img)

    # #ajustes morfologicos
    # dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)
    # dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
    
    dst = perspective_warp(dst, dst_size=(dst.shape[1], dst.shape[0]))
    
    out_img, curves, lanes, ploty = sliding_window(dst, margin=int(frame.shape[1]*.2))
    curverad=get_curve(img, curves[0],curves[1])
    img_ = draw_lanes(frame, curves[0], curves[1])
    
    # #midpoint 
    cv2.line(img_, (midpoint, int(frame.shape[0]*0.9)),(midpoint, int(frame.shape[0]*0.9)), red, 8)
        
    # print(np.asarray(curves).shape)
    # print(curverad)
    
    cv2.imshow("captura de imagen", frame)

    cv2.imshow("out", out_img)
    cv2.imshow("hola",dst)
    cv2.imshow("image", img_)

    img_msg = CvBridge().cv2_to_imgmsg(img_)

    filtered_image.publish(img_msg, "RGB8")

    cv2.waitKey(1)