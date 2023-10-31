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
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
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
#from misc import make_folder, time_synchronized
#from model_utils import create_model
#from kitti_data_utils import gen_hm_radius, compute_radius, Calibration, get_filtered_lidar
#from kitti_bev_utils import makeBEVMap, drawRotatedBox, get_corners
#from torch_utils import _sigmoid
#from evaluation_utils import decode, post_processing, draw_predictions, convert_det_to_real_values
import cv2

from models import resnet, fpn_resnet

BEV_HEIGHT = 800

BEV_WIDTH = 800

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # or os.makedirs(folder_name, exist_ok=True)

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def create_model(configs):
    """Create model based on architecture name"""
    try:
        arch_parts = configs.arch.split('_')
        num_layers = int(arch_parts[-1])
    except:
        raise ValueError
    if 'fpn_resnet' in configs.arch:
        print('using ResNet architecture with feature pyramid')
        model = fpn_resnet.get_pose_net(num_layers=num_layers, heads=configs.heads, head_conv=configs.head_conv,
                                        imagenet_pretrained=configs.imagenet_pretrained)
    elif 'resnet' in configs.arch:
        print('using ResNet architecture')
        model = resnet.get_pose_net(num_layers=num_layers, heads=configs.heads, head_conv=configs.head_conv,
                                    imagenet_pretrained=configs.imagenet_pretrained)
    else:
        assert False, 'Undefined model backbone'

    return model

def get_filtered_lidar(lidar, boundary, labels=None):
    minX = boundary['minX']
    maxX = boundary['maxX']
    minY = boundary['minY']
    maxY = boundary['maxY']
    minZ = boundary['minZ']
    maxZ = boundary['maxZ']

    # Remove the point out of range x,y,z
    mask = np.where((lidar[:, 0] >= minX) & (lidar[:, 0] <= maxX) &
                    (lidar[:, 1] >= minY) & (lidar[:, 1] <= maxY) &
                    (lidar[:, 2] >= minZ) & (lidar[:, 2] <= maxZ))
    lidar = lidar[mask]
    lidar[:, 2] = lidar[:, 2] - minZ

    if labels is not None:
        label_x = (labels[:, 1] >= minX) & (labels[:, 1] < maxX)
        label_y = (labels[:, 2] >= minY) & (labels[:, 2] < maxY)
        label_z = (labels[:, 3] >= minZ) & (labels[:, 3] < maxZ)
        mask_label = label_x & label_y & label_z
        labels = labels[mask_label]
        return lidar, labels
    else:
        return lidar
    
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

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (torch.floor_divide(topk_inds, width)).float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (torch.floor_divide(topk_ind, K)).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def _sigmoid(x):
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)


def decode(hm_cen, cen_offset, direction, z_coor, dim, K=40):
    batch_size, num_classes, height, width = hm_cen.size()

    hm_cen = _nms(hm_cen)
    scores, inds, clses, ys, xs = _topk(hm_cen, K=K)
    if cen_offset is not None:
        cen_offset = _transpose_and_gather_feat(cen_offset, inds)
        cen_offset = cen_offset.view(batch_size, K, 2)
        xs = xs.view(batch_size, K, 1) + cen_offset[:, :, 0:1]
        ys = ys.view(batch_size, K, 1) + cen_offset[:, :, 1:2]
    else:
        xs = xs.view(batch_size, K, 1) + 0.5
        ys = ys.view(batch_size, K, 1) + 0.5

    direction = _transpose_and_gather_feat(direction, inds)
    direction = direction.view(batch_size, K, 2)
    z_coor = _transpose_and_gather_feat(z_coor, inds)
    z_coor = z_coor.view(batch_size, K, 1)
    dim = _transpose_and_gather_feat(dim, inds)
    dim = dim.view(batch_size, K, 3)
    clses = clses.view(batch_size, K, 1).float()
    scores = scores.view(batch_size, K, 1)

    # (scores x 1, ys x 1, xs x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
    # (scores-0:1, ys-1:2, xs-2:3, z_coor-3:4, dim-4:7, direction-7:9, clses-9:10)
    # detections: [batch_size, K, 10]
    detections = torch.cat([scores, xs, ys, z_coor, dim, direction, clses], dim=2)

    return detections

import torch.nn.functional as F

def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()

    return heat * keep

boundary = {
    "minX": -100,
    "maxX": 100,
    "minY": -100,
    "maxY": 100,
    "minZ": -2.73,
    "maxZ": 1.27
}

bound_size_x = boundary['maxX'] - boundary['minX']
bound_size_y = boundary['maxY'] - boundary['minY']
bound_size_z = boundary['maxZ'] - boundary['minZ']

def post_processing(detections, num_classes=3, down_ratio=4, peak_thresh=0.60 ):
    """
    :param detections: [batch_size, K, 10]
    # (scores x 1, xs x 1, ys x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
    # (scores-0:1, xs-1:2, ys-2:3, z_coor-3:4, dim-4:7, direction-7:9, clses-9:10)
    :return:
    """
    # TODO: Need to consider rescale to the original scale: x, y

    ret = []
    for i in range(detections.shape[0]):
        top_preds = {}
        classes = detections[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            # x, y, z, h, w, l, yaw
            top_preds[j] = np.concatenate([
                detections[i, inds, 0:1],
                detections[i, inds, 1:2] * down_ratio,
                detections[i, inds, 2:3] * down_ratio,
                detections[i, inds, 3:4],
                detections[i, inds, 4:5],
                detections[i, inds, 5:6] / bound_size_y * BEV_WIDTH,
                detections[i, inds, 6:7] / bound_size_x * BEV_HEIGHT,
                get_yaw(detections[i, inds, 7:9]).astype(np.float32)], axis=1)
                
            if len(top_preds[j]) > 5:
                list_of_Prediction_of_SFA_3D_Model = list( top_preds[ j ][ : , 0 ] )
                list_of_Prediction_of_SFA_3D_Model = sorted( list_of_Prediction_of_SFA_3D_Model , reverse = True )
                keep_inds = (top_preds[j][:, 0] > list_of_Prediction_of_SFA_3D_Model[ 5 ] )
                top_preds[j] = top_preds[j][keep_inds] 
                
            # Filter by peak_thresh
            if len(top_preds[j]) > 0:
                keep_inds = (top_preds[j][:, 0] > peak_thresh)
                top_preds[j] = top_preds[j][keep_inds]

                           
            
        ret.append(top_preds)

    return ret

def get_yaw(direction):
    return np.arctan2(direction[:, 0:1], direction[:, 1:2])

def convert_det_to_real_values(detections, num_classes=3):
    kitti_dets = []
    for cls_id in range(num_classes):
        if len(detections[cls_id]) > 0:
            for det in detections[cls_id]:
                # (scores-0:1, x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
                _score, _x, _y, _z, _h, _w, _l, _yaw = det
                _yaw = -_yaw
                x = _y / BEV_HEIGHT * bound_size_x + boundary['minX']
                y = _x / BEV_WIDTH * bound_size_y + boundary['minY']
                z = _z + boundary['minZ']
                w = _w / BEV_WIDTH * bound_size_y
                l = _l / BEV_HEIGHT * bound_size_x

                kitti_dets.append([cls_id, x, y, z, _h, w, l, _yaw])

    return np.array(kitti_dets)

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
                        default='/home/ofel04/SFA3D/checkpoints/Fine_Tuning_using_SFA_3D_Dataset_using_Racing_Scene/Model_Fine_Tuning_using_SFA_3D_Dataset_using_Racing_Scene_epoch_398.pth', metavar='PATH',
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
    parser.add_argument('--peak_thresh', type=float, default=0.1)
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_fpn_resnet_18', metavar='PATH',
                        help='the video filename if the output format is video')
    parser.add_argument('--output-width', type=int, default=800,
                        help='the width of showing output, the height maybe vary')

    configs = EasyDict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False  # For testing on 1 GPU only

    configs.input_size = (800, 800)
    configs.hm_size = (200, 200)
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
    "minX": -100,
    "maxX": 100,
    "minY": -100,
    "maxY": 100,
    "minZ": -2.73,
    "maxZ": 1.27
}

    configs.bev_width = 800
    configs.bev_height = 800
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

colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0], [255, 120, 0],
          [255, 120, 120], [0, 120, 0], [120, 255, 255], [120, 0, 255]]

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
        Height = 800 + 1
        Width = 800 + 1

        DISCRETIZATION = ( 100 - ( -100 )) / 800

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
        RGB_Map[2, :, :] = densityMap[:BEV_HEIGHT, :BEV_WIDTH]  # r_map
        RGB_Map[1, :, :] = heightMap[:BEV_HEIGHT, :BEV_WIDTH]  # g_map
        RGB_Map[0, :, :] = intensityMap[:BEV_HEIGHT, :BEV_WIDTH]  # b_map

        return RGB_Map
    
    def draw_predictions( self , img_path , img, detections, num_classes=3):

        #print( "Object Detection in SFA 3D Object Detections are : " + str( detections ) )

        
        for j in range(num_classes):
            if len(detections[j]) > 0:

                i = 0
                for det in detections[j]:

                    i = i + 1
                    # (scores-0:1, x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
                    _score, _x, _y, _z, _h, _w, _l, _yaw = det
                    bev_corners = get_corners(_x, _y, _w, _l, _yaw)
                    self.drawRotatedBox(img_path, img, _x, _y, _w, _l, _yaw, colors[int(j)], writing_mode="Prediction" , confidence_score_prediction= _score )

        return img

    def drawRotatedBox( self , img_path, img, x, y, w, l, yaw, color , writing_mode = None , confidence_score_prediction = 100 ):
        #x = ( x + img.shape[ 0 ]/2)%img.shape[ 0 ]
        bev_corners = get_corners(x, y, w, l, yaw)
        #print(bev_corners)
        #print( "Image Shape is : " + str( img.shape ))
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
        time_to_Make_Filtered_LiDAR_Data = time.time()
        bev_map = self.makeBEVMap(filtered_lidar, configs.boundary)
        bev_map = torch.from_numpy(bev_map)
        bev_map = torch.unsqueeze(bev_map, dim=0)
        input_bev_maps = bev_map.to(configs.device, non_blocking=True).float()
        print( "Time to Have Preprocessing LiDAR Data is {}".format( time.time() - time_to_Make_Filtered_LiDAR_Data ))
        #t1 = time_synchronized()
        outputs = model(input_bev_maps)
        outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
        outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
        # detections size (batch_size, K, 10)
        detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                            outputs['dim'], K=configs.K)
        detections = detections.detach().cpu().numpy().astype(np.float32)
        
        detections = post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)
        #t2 = time_synchronized()

        #print( "Time of Object Detection using SFA3D Dataset is : " + str( ( t2 - t1 )* 1000 ))

        detections = detections[0]  # only first batch
        # Draw prediction in the image

        # Draw prediction in the image
        #bev_map = (bev_map.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        #bev_map = cv2.resize(bev_map, (1216, 1216))
        #bev_map = self.draw_predictions( str( t1 ) , bev_map, detections.copy(), configs.num_classes )

        #detections_of_sfa_3d = np.array( detections.copy() )

        #print( "Shape of Detection SFA 3D Object Detection is : " + str( detections_of_sfa_3d.shape ))

        #detections_of_sfa_3d = np.squeeze( detections_of_sfa_3d , axis = 0 )


        # Rotate the bev_map
        #bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
        
        final_dets = convert_det_to_real_values(detections)
        print("final_dets")
        print(final_dets)
        #t2 = time_synchronized()
        #t_f = time.time()
        #print("inference_time", t_f - t_t)

        #exit()

        ### Here to change 

        try :
            boxes_lidar = final_dets[ : , 1 : 8 ]#outputs["box3d_lidar"].detach().cpu().numpy()
            #print("  predict boxes:", boxes_lidar.shape)
        except :
            boxes_lidar = np.array( [] )
            #print( "There is no Object Detection SFA 3D Object Detection...")

        
        try :
            scores = final_dets[ : , 0 ]#outputs["scores"].detach().cpu().numpy()
        except :
            scores = np.array( [] )

        types = [ "Cars" for i in range( len( scores ) )]#outputs["label_preds"].detach().cpu().numpy()

        return bev_map , scores, boxes_lidar, types

def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
    '''

    '''

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
    arr_bbox = BoundingBoxArray() #carmaker

    #print( "LiDAR Object Detection Message Header is : " + str( msg.header ))

    msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    np_p = get_xyz_points(msg_cloud, True)
    np_p[:, 3] = np_p[:, 3] / np.max(np_p[:, 3]) * 10
    time_of_Predict_SFA_3D_Object_Detection = time.time()
    bev_image , scores, dt_box_lidar, types = proc_1.run(np_p)
    print( "Time to make Prediction SFA 3D Bounding Box is : {0:.4f}".format( time.time() - time_of_Predict_SFA_3D_Object_Detection ))
    if scores.size != 0:
        for i in range(scores.size):
            if math.sqrt(dt_box_lidar[i][0]**2 + dt_box_lidar[i][1]**2) < 0.707:
                continue
            else:
                bbox = BoundingBox() #carmaker  
                bbox.header.frame_id = msg.header.frame_id
                bbox.header.stamp = msg.header.stamp#rospy.Time.now()
                q = quaternion_from_euler(0, 0, float(dt_box_lidar[i][6]))
                bbox.pose.orientation.x = (q[0]) #1
                bbox.pose.orientation.y = (q[1]) #2
                bbox.pose.orientation.z = -q[2] #3
                bbox.pose.orientation.w = -q[3] #0
                bbox.pose.position.x = float(dt_box_lidar[i][0])
                bbox.pose.position.y = float(dt_box_lidar[i][1])
                bbox.pose.position.z = float(dt_box_lidar[i][2])
                bbox.dimensions.x = float(dt_box_lidar[i][5])
                bbox.dimensions.y = float(dt_box_lidar[i][4])
                bbox.dimensions.z = float(dt_box_lidar[i][3])
                bbox.value = scores[i]
                bbox.label = 1#types[i]
                arr_bbox.boxes.append(bbox)

    total_inference_time = round( float( time.time() - t_t ) , 5 )

    total_number_of_SFA_3D_Prediction_per_Seconds = round( 1 / total_inference_time , 3 )

    print("total callback time: {} seconds \nAnd Total SFA 3D Prediction per Second : {} Detection Per Seconds".format( total_inference_time  , total_number_of_SFA_3D_Prediction_per_Seconds ))

    arr_bbox.header.frame_id = msg.header.frame_id
    arr_bbox.header.stamp.secs = msg.header.stamp.secs
    arr_bbox.header.stamp.nsecs= msg.header.stamp.nsecs

    print( "Boxes SFA 3D Object Detection Result is : " + str( arr_bbox.boxes ))
    if len(arr_bbox.boxes) > 0:
        pub_arr_bbox.publish(arr_bbox)
        arr_bbox.boxes = []
    else:
        arr_bbox.boxes = []
        pub_arr_bbox.publish(arr_bbox)

    #print( "Finish make LiDAR SFA3D Object Detection :-) wkwkww..")
   
if __name__ == "__main__":
    configs = parse_test_configs()

    model_path = "/home/ofel04/Model_SFA_3D_Dataset_Input_800_Trained_Duplicate_Overtake_Scene_Epoch_298.pth"#"/home/ofel04/Model_SFA_3D_Dataset_Input_800_Epoch_300.pth"#/home/ofel04/SFA3D/checkpoints/Training_SFA_3D_Hyundai_Race_Rosebag_Using_Input_1000_Try/Loss_tensor(0.1878, device='cuda:0')_Model_Training_SFA_3D_Hyundai_Race_Rosebag_Using_Input_1000_Try_epoch_264.pth"#"../checkpoints/Training_SFA_3D_using_Hyundai_Race_with_Small_LiDAR_Object_Detection/Loss_tensor(0.1473, device='cuda:0')_Model_Training_SFA_3D_using_Hyundai_Race_with_Small_LiDAR_Object_Detection_epoch_600.pth"#'/home/ofel04/SFA3D/checkpoints/Fine_Tuning_using_SFA_3D_Dataset_using_Racing_Scene/Model_Fine_Tuning_using_SFA_3D_Dataset_using_Racing_Scene_epoch_398.pth'

    try :
        configs.pretrained_path = model_path
    except :
        print( "Using SFA 3D Model Object Detection : " + configs.pretrained_path )

    model = create_model(configs)
    
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)

    out_cap = None

    model.eval()
    global proc
    #model_path = "../checkpoints/Training_SFA_3D_using_Hyundai_Race_with_Small_LiDAR_Object_Detection_Try_Small_LiDAR_Object_Detection_800_Points/Loss_tensor(0.1207, device='cuda:0')_Model_Training_SFA_3D_using_Hyundai_Race_with_Small_LiDAR_Object_Detection_Try_Small_LiDAR_Object_Detection_800_Points_epoch_300.pth"#'/home/ofel04/SFA3D/checkpoints/Fine_Tuning_using_SFA_3D_Dataset_using_Racing_Scene/Model_Fine_Tuning_using_SFA_3D_Dataset_using_Racing_Scene_epoch_398.pth'

    proc_1 = Processor_ROS(model_path)

    #proc_1 = Processor_ROS()
    
    # proc_1.initialize()
    
    rospy.init_node('sfa3d_ros_node')
    # sub_lidar_topic = [ "/CarMaker/Sensor/Lidar_64ch_1"]
    sub_lidar_topic = [ "/ouster/points"]
    
    sub_ = rospy.Subscriber(sub_lidar_topic[0], PointCloud2, rslidar_callback, queue_size=1, buff_size=2**24)
    
    # pub_arr_bbox = rospy.Publisher("pp_boxes", BoundingBoxArray, queue_size=1)
    pub_arr_bbox = rospy.Publisher("pp_boxes", BoundingBoxArray, queue_size=1) #carmaker

    print("[+] se_ssd ros_node has started!")    
    rospy.spin()
