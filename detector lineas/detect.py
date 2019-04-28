from functions.utils import *
import numpy as np
# import pandas as pd
import cv2
# import os
# import glob
from matplotlib.pyplot import subplots as plt
from matplotlib.pyplot import subplots_adjust as plta
# import pickle


red = (0, 0, 255)
blue = (255, 0, 0)
green = (0, 255, 0)
camara = "challenge_video.mp4"
cap = cv2.VideoCapture(camara)

while(True):

    # lectura de camara
    ret, frame = cap.read()
    
    if not ret:
        print( "Error en la lectura de imagen" )
        cap = cv2.VideoCapture(camara)
    
    midpoint = int(frame.shape[1]/2)

    # img = cv2.imread('Curved-Lane-Lines/test_images/test3.jpg')
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dst = pipeline(img)
    dst = perspective_warp(dst, dst_size=(1280,720))

    out_img, curves, lanes, ploty = sliding_window(dst)
    curverad=get_curve(img, curves[0],curves[1])
    img_ = draw_lanes(frame, curves[0], curves[1])
    
    #midpoint 
    cv2.line(img_, (midpoint, frame.shape[0]-100),(midpoint, frame.shape[0]-100), red, 8)
        
    # print(np.asarray(curves).shape)
    # print(curverad)
    
    cv2.imshow("out", out_img)
    cv2.imshow("hola",dst)
    cv2.imshow("image", img_)


    f, (ax1, ax2, ax3, ax4, ax5) = plt(1, 5, figsize=(100, 20))
    #f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original', fontsize=100)
    ax2.imshow(dst)
    ax2.set_title('Filter+Perspective Tform', fontsize=100)
    ax3.imshow(out_img)
    ax3.plot(curves[0], ploty, color='yellow', linewidth=30)
    ax3.plot(curves[1], ploty, color='yellow', linewidth=30)
    ax3.set_title('Sliding window+Curve Fit', fontsize=100)
    ax4.imshow(img_)

    ax4.set_title('Overlay Lanes', fontsize=100)
    plta(left=0., right=1, top=0.9, bottom=0.)

    cv2.waitKey(1)