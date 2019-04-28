from functions.utils import *
import numpy as np
# import pandas as pd
import cv2
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
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)
    dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
    
    dst = perspective_warp(dst, dst_size=(dst.shape[1], dst.shape[0]))

    out_img, curves, lanes, ploty = sliding_window(dst, margin=int(frame.shape[1]*.3))
    curverad=get_curve(img, curves[0],curves[1])
    img_ = draw_lanes(frame, curves[0], curves[1])
    
    # #midpoint 
    cv2.line(img_, (midpoint, int(frame.shape[0]*0.9)),(midpoint, int(frame.shape[0]*0.9)), red, 8)
        
    # print(np.asarray(curves).shape)
    # print(curverad)
    
    cv2.imshow("out", out_img)
    cv2.imshow("hola",dst)
    cv2.imshow("image", img_)

    cv2.waitKey(1)