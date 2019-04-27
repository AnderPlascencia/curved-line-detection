from functions.utils import *
import numpy as np
import pandas as pd
import cv2
import os
import glob
import matplotlib.pyplot as plt
import pickle

camara = "harder_challenge_video.mp4"
cap = cv2.VideoCapture(camara)

while(True):

    # lectura de camara
    ret, frame = cap.read()
    
    if not ret:
        print( "Error en la lectura de imagen" )
        cap = cv2.VideoCapture(camara)
    
    # img = cv2.imread('Curved-Lane-Lines/test_images/test3.jpg')
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dst = pipeline(img)
    dst = perspective_warp(dst, dst_size=(1280,720))

    out_img, curves, lanes, ploty = sliding_window(dst)
    curverad=get_curve(img, curves[0],curves[1])
    img_ = draw_lanes(frame, curves[0], curves[1])
    
    print(np.asarray(curves).shape)
    print(curverad)
    
    cv2.imshow("out", out_img)
    cv2.imshow("hola",dst)
    cv2.imshow("image", img_)

    cv2.waitKey(10)