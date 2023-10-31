#!/usr/bin/env python

import rospy
import ros_numpy
import numpy as np
import copy
import json
import os
import sys
import torch
import time 
import math
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
#from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from pyquaternion import Quaternion
from vision_msgs.msg import BoundingBox2DArray, BoundingBox3DArray, BoundingBox3D

"""
from det3d import torchie
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.core.anchor.anchor_generator import AnchorGeneratorRange
"""
from tf.transformations import quaternion_from_euler
import argparse
from easydict import EasyDict
from utils.misc import make_folder, time_synchronized
from models.model_utils import create_model
from data_process.kitti_data_utils import gen_hm_radius, compute_radius, Calibration, get_filtered_lidar
from data_process.kitti_bev_utils import makeBEVMap, drawRotatedBox, get_corners
from utils.torch_utils import _sigmoid
from utils.evaluation_utils import decode, post_processing, draw_predictions, convert_det_to_real_values
import cv2

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()



def parse_test_configs():
    parser = argparse.ArgumentParser(description='Testing config for the Implementation')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str,
                        default='Model_Fine_Tuning_using_SFA_3D_Dataset_using_Racing_Scene_epoch_398.pth', metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--K', type=int, default=50,
                        help='the number of top K')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')
    parser.add_argument('--peak_thresh', type=float, default=0.4)
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_fpn_resnet_18', metavar='PATH',
                        help='the video filename if the output format is video')
    parser.add_argument('--output-width', type=int, default=1216,
                        help='the width of showing output, the height maybe vary')

    configs = EasyDict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False  # For testing on 1 GPU only

    configs.input_size = (1216, 1216)
    configs.hm_size = (152, 152)
    configs.down_ratio = 4
    configs.max_objects = 50

    configs.imagenet_pretrained = False
    configs.head_conv = 64
    configs.num_classes = 3
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2  # sin, cos

    configs.heads = {
        'hm_cen': configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction': configs.num_direction,
        'z_coor': configs.num_z,
        'dim': configs.num_dim
    }
    configs.num_input_features = 4

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.root_dir = './'
    configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'kitti')

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.root_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

    configs.boundary = {
    "minX": 0,
    "maxX": 50,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}

    configs.bev_width = 608
    configs.bev_height = 608
    configs.discretization = (configs.boundary["maxX"] - configs.boundary["minX"]) / configs.bev_height

    return configs

def yaw2quaternion(yaw: float) -> Quaternion:
    return Quaternion(axis=[0,0,1], radians=yaw)

def get_annotations_indices(types, thresh, label_preds, scores):
    indexs = []
    annotation_indices = []
    for i in range(label_preds.shape[0]):
        if label_preds[i] == types:
            indexs.append(i)
    for index in indexs:
        if scores[index] >= thresh:
            annotation_indices.append(index)
    return annotation_indices  


def remove_low_score_nu(image_anno, thresh):
    img_filtered_annotations = {}
    label_preds_ = image_anno["label_preds"].detach().cpu().numpy()
    scores_ = image_anno["scores"].detach().cpu().numpy()
    
    car_indices =                  get_annotations_indices(0, 0.15, label_preds_, scores_)
    truck_indices =                get_annotations_indices(1, 1, label_preds_, scores_)
    construction_vehicle_indices = get_annotations_indices(2, 1, label_preds_, scores_)
    bus_indices =                  get_annotations_indices(3, 1, label_preds_, scores_)
    trailer_indices =              get_annotations_indices(4, 1, label_preds_, scores_)
    barrier_indices =              get_annotations_indices(5, 1, label_preds_, scores_)
    motorcycle_indices =           get_annotations_indices(6, 1, label_preds_, scores_)
    bicycle_indices =              get_annotations_indices(7, 1, label_preds_, scores_)
    pedestrain_indices =           get_annotations_indices(8, 1, label_preds_, scores_)
    traffic_cone_indices =         get_annotations_indices(9, 1, label_preds_, scores_)
    
    for key in image_anno.keys():
        if key == 'metadata':
            continue
        img_filtered_annotations[key] = (
            image_anno[key][car_indices +
                            pedestrain_indices + 
                            bicycle_indices +
                            bus_indices +
                            construction_vehicle_indices +
                            traffic_cone_indices +
                            trailer_indices +
                            barrier_indices +
                            truck_indices
                            ])

    return img_filtered_annotations


class Processor_ROS:
    def __init__(self, model_path):
        self.points = None
        self.object_detection_sfa_3d_result = None
        self.model_path = model_path
        self.device = None
        self.net = None
        self.voxel_generator = None
        self.inputs = None

    def makeBEVMap( self , PointCloud_, boundary = { "maxZ" : 1.27 , "minZ" : -2.73 }):
        Height = 1216 + 1
        Width = 1216 + 1

        DISCRETIZATION = ( 100 - ( -100 )) / 1216

        # Discretize Feature Map
        PointCloud = np.copy(PointCloud_)
        PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / DISCRETIZATION + Height/ 2 )) ### ofel changed
        PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / DISCRETIZATION) + Width / 2) ### ofel changed

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

        return RGB_Map
    
    def draw_predictions( self , img_path , img, detections, num_classes=3):

        print( "Object Detection in SFA 3D Object Detections are : " + str( detections ) )

        #f = open( "Output_Test_Dataset_SFA_3D_from_Bag.txt" , "a+" )

        #print( 'meta', metadatas )

        #Name_of_Point_Cloud_Dataset = metadatas[ "img_path" ].split( "/" )[2]
        
        for j in range(num_classes):
            if len(detections[j]) > 0:

                i = 0
                for det in detections[j]:

                    i = i + 1
                    # (scores-0:1, x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
                    _score, _x, _y, _z, _h, _w, _l, _yaw = det
                    bev_corners = get_corners(_x, _y, _w, _l, _yaw)
                    self.drawRotatedBox(img_path, img, _x, _y, _w, _l, _yaw, cnf.colors[int(j)], writing_mode="Prediction" , confidence_score_prediction= _score )

        return img

    def drawRotatedBox( self , img_path, img, x, y, w, l, yaw, color , writing_mode = None , confidence_score_prediction = 100 ):
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


    def run(self, points):
        t_t = time.time()
        # print(f"input points shape: {points.shape}")
        num_features = 4        
        self.points = points.reshape([-1, num_features])
        filtered_lidar = get_filtered_lidar(self.points, configs.boundary)
        bev_map = self.makeBEVMap(filtered_lidar, configs.boundary)
        bev_map = torch.from_numpy(bev_map)
        bev_map = torch.unsqueeze(bev_map, dim=0)
        input_bev_maps = bev_map.to(configs.device, non_blocking=True).float()
        t1 = time_synchronized()
        outputs = model(input_bev_maps)
        outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
        outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
        # detections size (batch_size, K, 10)
        detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                            outputs['dim'], K=configs.K)
        detections = detections.detach().cpu().numpy().astype(np.float32)
        
        detections = post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)
        t2 = time_synchronized()

    #4D Radar

        print( "Time of Object Detection using SFA3D Dataset is : " + str( ( t2 - t1 )* 1000 ))

        detections = detections[0]  # only first batch
        # Draw prediction in the image

        # Draw prediction in the image
        bev_map = (bev_map.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        bev_map = cv2.resize(bev_map, (1216, 1216))
        bev_map = self.draw_predictions( str( t1 ) , bev_map, detections.copy(), configs.num_classes )

        detections_of_sfa_3d = np.array( detections.copy() )

        detections_of_sfa_3d = np.squeeze( detections_of_sfa_3d , axis = 0 )


        # Rotate the bev_map
        bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
        
        final_dets = convert_det_to_real_values(detections)
        print("final_dets")
        print(final_dets)
        t2 = time_synchronized()
        t_f = time.time()
        print("inference_time", t_f - t_t)

        #exit()

        ### Here to change 

        boxes_lidar = detections_of_sfa_3d[ : , 1 : 8 ]#outputs["box3d_lidar"].detach().cpu().numpy()
        print("  predict boxes:", boxes_lidar.shape)

        scores = detections_of_sfa_3d[ : , 0 ]#outputs["scores"].detach().cpu().numpy()
        types = [ "Cars" for i in range( len( scores ) )]#outputs["label_preds"].detach().cpu().numpy()

        return bev_map , scores, boxes_lidar, types

        #return final_dets

def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
    '''

    '''
    # print("reflectivity")
    # print(cloud_array['reflectivity'])

    # print("intensity")
    # print(cloud_array['intensity'])

    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z']) \
        & np.isfinite(cloud_array['intensity'])
        cloud_array = cloud_array[mask]
    points = np.zeros(cloud_array.shape + (4,), dtype=dtype)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']
    points[...,3] = cloud_array['intensity']
    return points

def xyz_array_to_pointcloud2(points_sum, stamp=None, frame_id=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array of points.
    '''
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points_sum.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
        # PointField('i', 12, PointField.FLOAT32, 1)
        ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = points_sum.shape[0]
    msg.is_dense = int(np.isfinite(points_sum).all())
    msg.data = np.asarray(points_sum, np.float32).tostring()
    return msg

def rslidar_callback(msg):
    t_t = time.time()
    arr_bbox = BoundingBox3DArray() #carmaker

    msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    np_p = get_xyz_points(msg_cloud, True)
    np_p[:, 3] = np_p[:, 3] / np.max(np_p[:, 3]) * 10
    bev_image , scores, dt_box_lidar, types = proc_1.run(np_p)
    if scores.size != 0:
        for i in range(scores.size):
            if math.sqrt(dt_box_lidar[i][0]**2 + dt_box_lidar[i][1]**2) < 0.707:
                continue
            else:
                # arr_bbox.header.frame_id = msg.header.frame_id
                # arr_bbox.header.stamp = rospy.Time.now()
                # print(dt_box_lidar[i])
                bbox = BoundingBox3D() #carmaker
                q = quaternion_from_euler(0, 0, float(dt_box_lidar[i][6]))
                bbox.center.orientation.x = (q[0])
                bbox.center.orientation.y = (q[1])
                bbox.center.orientation.z = - q[2]
                bbox.center.orientation.w = - q[3]
                bbox.center.position.x = float(dt_box_lidar[i][0]) 
                bbox.center.position.y = float(dt_box_lidar[i][1]) 
                bbox.center.position.z = float(dt_box_lidar[i][2])
                bbox.size.x = float(dt_box_lidar[i][4])
                bbox.size.y = float(dt_box_lidar[i][3])
                bbox.size.z = float(dt_box_lidar[i][5])
                # bbox.value = scores[i]
                # bbox.label = int(types[i])
                arr_bbox.boxes.append(bbox)
    print("total callback time: ", time.time() - t_t)
    arr_bbox.header.frame_id = "Fr1A"
    arr_bbox.header.stamp = msg.header.stamp
    if len(arr_bbox.boxes) > 0:
        pub_arr_bbox.publish(arr_bbox)
        arr_bbox.boxes = []
    else:
        arr_bbox.boxes = []
        pub_arr_bbox.publish(arr_bbox)
   
if __name__ == "__main__":
    configs = parse_test_configs()
    model = create_model(configs)
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)

    out_cap = None

    model.eval()
    global proc
    model_path = './checkpoints/fpn_resnet_18/Model_fpn_resnet_18_epoch_200.pth'

    proc_1 = Processor_ROS(model_path)
    
    # proc_1.initialize()
    
    rospy.init_node('sfa3d_ros_node')
    # sub_lidar_topic = [ "/CarMaker/Sensor/Lidar_64ch_1"]
    sub_lidar_topic = [ "/ouster/points"]
    
    sub_ = rospy.Subscriber(sub_lidar_topic[0], PointCloud2, rslidar_callback, queue_size=1, buff_size=2**24)
    
    # pub_arr_bbox = rospy.Publisher("pp_boxes", BoundingBoxArray, queue_size=1)
    pub_arr_bbox = rospy.Publisher("pp_boxes", BoundingBox3DArray, queue_size=1) #carmaker

    print("[+] se_ssd ros_node has started!")    
    rospy.spin()