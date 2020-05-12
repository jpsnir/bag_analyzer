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
import logging
import rosbag
import argparse
import yaml

class BagAnalyser:
    '''
    analyses a bag/a directory of images from a given  that contains images and imu data 
    - Optical flow analysis (sparse and dense) from camera images at a particular resolution
    - Motion analysis from IMU
    -
    '''
    def __init__(self,bag_path):
        self.logger = None
        self.node = rospy.init_node(name,anonymous=True,log_level=rospy.INFO)
        self.bag = None
        self.optical_flow_figure = None 
        self.imu_motion_figure = None 
        self.optical_flow_data = None
        self.imu_data = None
        self.image_sizes = [] 
        
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
        logger.debug('Optical Flow analysis started')
        if category == 'dense':
            logger.debug('Dense optical flow')
            self.dense_optical_flow()
        elif category == 'sparse':
            logger.debug('Sparse optical flow')
        
    def analyse_imu(se):
        logger.debug(' analysis started')
        
    def dense_optical_flow(self):
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
    
def configure_logger(name):
    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(asctime)s - \
                                  %(name)s -\
                                  %(levelname)s -\
                                  %(message)s')
    logger.setLevel(logging.INFO)
    
    # Console logger
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    
    # file logger
    fh = logging.FileHandler()
    fh.setLevel(logging.INFO)
    
    # set formatter for both 
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(sh)
    
    return logger
    
def configure_argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Reads a rosbag containing\
                                     different sensors data(IMU and camera).\
                                     For images only, just add the folder of\
                                     images.\n \n')
                                     
    parser.add_argument('-i',
                        '--input', 
                        help='Input rosbag file to input',
                        default='/home/jagat/datasets/ \
                        vio_rosbags/euroc_datasets/MH_05.bag')
    
    parser.add_argument('-c',
                        '--config_file', 
                        help='Yaml file which specifies\
                        the image and imu topics',
                        default = '../config/bag_analysis_config.yaml')
    
    parser.add_argument('-log',
                        '--loglevel',
                        help='defines the log level',
                        default=logging.INFO)     
    return parser

def load_yaml_file(filename, config=None):
    with open(args.config_file,'r') as f:
        config = yaml.safe_load(f)
        return config

if __name__=='__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()
    name = 'Bag_analyzer'
    # Initialise ros nodes
    config_parameters = load_yaml_file(args.config)
    ba = BagAnalyser(args.input)
    ba.logger = configure_logger(name)  
    
    
    ba.logger.info('Details of config file \n')
    for key in config.keys():
        print(key+':'+str(config[key]))
    raw_input('\n Press Enter to continue')
    
    
  
    
    