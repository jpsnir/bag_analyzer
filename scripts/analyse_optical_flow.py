#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:04:15 2020

@author: jagat
"""

import rospy
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import logging as lg
import rosbag
import argparse

class BagAnalyser:
    '''
    analyses a bag/a directory of images from a given  that contains images and imu data 
    - Optical flow analysis (sparse and dense) from camera images at a particular resolution
    - Motion analysis from IMU
    -
    '''
    def __init__(self,bag_path,):
        lg.debug('Initialise');
        self.bag = None
        self.optical_flow_figure = None 
        self.imu_motion_figure = None 
        self.optical_flow_data = None
        self.imu_data = None
        self.image_size = np.array([2,1]) 
        
    def analyse_optical_flow(self,category=None,):
        '''
        computes dense/sparse optical flow of the images of a specific topic in a bag file
        and plot the results
        Inputs:
            category - string input ('dense' or 'sparse')
            takes in the type of optical flow to be computed
        Outputs:
            returns the optical flow data and plots it as well
        '''
        lg.debug('Optical Flow analysis started')
        if category == 'dense':
            lg.debug('Dense optical flow')
            dense_optical_flow()
        elif category == 'sparse':
            lg.debug('Sparse optical flow')
        
    def analyse_imu(se):
        lg.debug(' analysis started')
        
    def dense_optical_flow():
        cap = cv.VideoCapture(cv.samples.findFile("vtest.avi"))
        ret, frame1 = cap.read()
        prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255
        while(1):
            ret, frame2 = cap.read()
            next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
            bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
            cv.imshow('frame2',bgr)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
            elif k == ord('s'):
                cv.imwrite('opticalfb.png',frame2)
                cv.imwrite('opticalhsv.png',bgr)
            prvs = next
                
if name=='__main__':
    