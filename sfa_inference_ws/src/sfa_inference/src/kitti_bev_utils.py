"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
"""

import math
import os
import sys

import cv2
import numpy as np

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import config.kitti_config as cnf


def makeBEVMap(PointCloud_, boundary):
    Height = cnf.BEV_HEIGHT + 1
    Width = cnf.BEV_WIDTH + 1

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / cnf.DISCRETIZATION + Height/ 2 )) ### ofel changed
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / cnf.DISCRETIZATION) + Width / 2) ### ofel changed

    # sort-3times
    sorted_indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[sorted_indices]
    _, unique_indices, unique_counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[unique_indices]

    # Height Map, Intensity Map & Density Map
    heightMap = np.zeros((Height, Width))
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    # some important problem is image coordinate is (y,x), not (x,y)
    max_height = float(np.abs(boundary['maxZ'] - boundary['minZ']))
    heightMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 2] / max_height

    normalizedCounts = np.minimum(1.0, np.log(unique_counts + 1) / np.log(64))
    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

    RGB_Map = np.zeros((3, Height - 1, Width - 1))
    RGB_Map[2, :, :] = densityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # r_map
    RGB_Map[1, :, :] = heightMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # g_map
    RGB_Map[0, :, :] = intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # b_map

    Upper_Visualization_of_RGB_Map_Bounding_Box_Prediction = RGB_Map[ : , : 608 , : ]

    """

    if RGB_Map.shape[ 1 ] == 1216 :

        print( "Make Visualization of Bounding Box SFA 3D Prediction...")

        RGB_Map[ : , : 608 , : ] = RGB_Map[ : , 608 : , : ]

        RGB_Map[ : , 608 : , : ] = Upper_Visualization_of_RGB_Map_Bounding_Box_Prediction

    print( RGB_Map.shape )

    """

    return RGB_Map


# bev image coordinates format
def get_corners(x, y, w, l, yaw):
    bev_corners = np.zeros((4, 2), dtype=np.float32)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    # front left
    bev_corners[0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw

    # rear left
    bev_corners[1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw

    # rear right
    bev_corners[2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw

    # front right
    bev_corners[3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw

    return bev_corners


def drawRotatedBox(img_path, img, x, y, w, l, yaw, color , writing_mode = None , confidence_score_prediction = 100 ):
    #x = ( x + img.shape[ 0 ]/2)%img.shape[ 0 ]
    bev_corners = get_corners(x, y, w, l, yaw)
    print(bev_corners)
    print( "Image Shape is : " + str( img.shape ))
    corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
    cv2.polylines(img, [corners_int], True, color, 2)
    corners_int = bev_corners.reshape(-1, 2).astype(int)
    
    bev_corners = bev_corners.astype( int )

    f = open( "Output_" + writing_mode + "Ground_Truth_Dataset_SFA_3D_from_Bag.txt" , "a+")
    f.write( "{} {} {} {} {} {} {} {} {} {}\n".format( img_path  ,  confidence_score_prediction , bev_corners[0, 0], bev_corners[0, 1], bev_corners[1, 0], bev_corners[1, 1] , bev_corners[ 2 ][ 0 ] , bev_corners[ 2 ][ 1 ] , bev_corners[ 3 ][ 0 ] , bev_corners[ 3 ][ 1 ]) )
    cv2.line(img, (corners_int[0, 0], corners_int[0, 1]), (corners_int[3, 0], corners_int[3, 1]), (255, 255, 0), 2)
    cv2.line(img, (-100, -100), (100, 100), (255, 0, 0), 2)

