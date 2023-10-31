"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Goenawan Christofel Rio
# DoC: 2023.03.17
# AI & Analytics Consultant at Hyundai Company
# AI & Robotics Researcher at Korea Advanced Institute of Science and Technology
# email: christofel.goenawan@kaist.ac.kr
-----------------------------------------------------------------------------------
# Description: Testing script
"""

import argparse
import sys
import os
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from easydict import EasyDict as edict
import cv2
import torch
import numpy as np

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import data_process.kitti_dataset as kitti_dataset

from data_process.kitti_dataloader import create_test_dataloader , create_val_dataloader , create_train_dataloader
from models.model_utils import create_model
from utils.misc import make_folder, time_synchronized
from utils.evaluation_utils import decode, post_processing, draw_predictions, convert_det_to_real_values
from utils.torch_utils import _sigmoid
import config.kitti_config as cnf
from data_process.transformation import lidar_to_camera_box
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from data_process.kitti_data_utils import Calibration

import pandas as pd

from sklearn.metrics import average_precision_score
from shapely.geometry import Polygon

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



def bb_intersection_over_union(boxA, boxB):
    

    polygon = Polygon([(boxA[ 0 ], boxA[ 1 ]), (boxA[ 2 ], boxA[ 3 ]), (boxA[ 4 ], boxA[ 5 ]), (boxA[ 6 ], boxA[ 7 ] ) ] )
    other_polygon = Polygon([(boxB[ 0 ], boxB[ 1 ]), (boxB[ 2 ], boxB[ 3 ]), (boxB[ 4 ], boxB[ 5 ]), ( boxB[ 6 ], boxB[ 7 ])])
    intersection = polygon.intersection(other_polygon)
    #print(intersection.area)

    Area_of_Union_2_Bounding_Box = polygon.area + other_polygon.area - intersection.area

    iou = intersection.area / Area_of_Union_2_Bounding_Box

    # return the intersection over union value
    return iou



def parse_test_configs():
    parser = argparse.ArgumentParser(description='Testing config for the Implementation')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str,
                        default='../checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth', metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--K', type=int, default=50,
                        help='the number of top K')
    parser.add_argument('--no_cuda', default= True, action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='mini-batch size (default: 4)')
    parser.add_argument('--peak_thresh', type=float, default=0.2)
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_fpn_resnet_18', metavar='PATH',
                        help='the video filename if the output format is video')
    parser.add_argument('--output-width', type=int, default=800,
                        help='the width of showing output, the height maybe vary')
    
    parser.add_argument("--is_using_Output_Prediction" , type=bool , default= True,
                        help="Is Making Output Text File of SFA 3D Prediction Bounding Box or not...")
                        
    parser.add_argument("--dataset_Directory_to_Test_SFA_3D_Dataset" , type=bool , default= True,
                        help="Directorey of Test SFA 3D Dataset to TestIs Making Output Text File of SFA 3D Prediction Bounding Box or not...")
    
    parser.add_argument('--dataset_dir', type=str, default= os.path.join('../../', 'dataset', 'kitti'),
                        help='Dataset for SFA 3D Dataset Bounding Box Rosebag Hyundai')
    parser.add_argument('--is_Dataset_SFA_3D_Predicted_Bounding_Box', type=bool, default= True ,
                        help='Is Dataset SFA 3D Bounding Box Hyundai Rosebag Predicted Bounding Box or Not')  
    parser.add_argument('--is_Training_SFA_3D_Model_Without_Color', type=bool, default=False,
                        help='is Training SFA 3D Model Without Color')                           

    configs = edict(vars(parser.parse_args()))
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
    configs.root_dir = '../'
    #configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'kitti')

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.root_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)
    
    
    


    return configs


if __name__ == '__main__':

    configs = parse_test_configs()

    lidar_aug = None

    NAME_FOLDER_OF_SFA3D_MODEL_ROSBAG_HYUNDAI = "/home/ofel04/SFA3D/checkpoints/Training_SFA_3D_on_All_Hyundai_Rosbag_Dataset_with_Hyundai_New_Race_Input_800_Range_100_m_Higher_Learning_Rate/"#"/home/ofel04/SFA3D/checkpoints/Training_SFA_3D_Dataset_Hyundai_Rosbag_Labelling_Intern_with_New_Data_Input_800_Range_100_M/"

    LIST_SFA_3D_MODEL_ROSBAG_HYUNDAI_EVALUATED = [ "epoch" + "_" + str( i ) + ".pth" for i in range( 160 , 170 ) ]

    SFA_3D_Ground_Truth_Bounding_Box_Dictionary = { "Image_Path" : [] ,
                                                   "Confidence_Score_Prediction" : [] ,
                                                   "BEV_Corner" : [] }
    

    # Create Ground Truth SFA 3D Model Rosbag Hyundai

    dataset = kitti_dataset.KittiDataset(configs, mode='val', lidar_aug=lidar_aug, hflip_prob=0., num_samples=configs.num_samples)

    print( "Creating Ground Truth SFA 3D Rosbag Hyundai Dataset" )

    #print('\n\nPress n to see the next sample >>> Press Esc to quit...')

    #Name_File_of_Prediction_and_Ground_Truth_Output_Files = "Output_" + "Ground_Truth" + "Ground_Truth_Dataset_SFA_3D_from_Bag.txt"

    """
    if os.path.exists( Name_File_of_Prediction_and_Ground_Truth_Output_Files ):

        os.system( "sudo rm " + str( Name_File_of_Prediction_and_Ground_Truth_Output_Files ))

        os.system( "sudo touch " + str( Name_File_of_Prediction_and_Ground_Truth_Output_Files ) )

    f = open( "Output_Test_True_Detection_Dataset_SFA_3D_from_Bag.txt" , "a+" )
    """

    #print( 'meta', metadatas )

    configs.no_cuda= True
    
    configs.device = "cpu"    

    for idx in range(len(dataset)):
        bev_map, labels, img_rgb, img_path = dataset.draw_img_with_label(idx)
        #print( 'Making Dataset SFA 3D Labelling Rosbag Data : ' + str( idx ))
        calib = Calibration(img_path.replace(".png", ".txt").replace("image_2", "calib"))
        

        for box_idx, (cls_id, x, y, z, h, w, l, yaw) in enumerate(labels):
            # Draw rotated box
            yaw = -yaw
      
            y1 = int((x - cnf.boundary['minX']) / cnf.DISCRETIZATION)
            x1 = int((y - cnf.boundary['minY']) / cnf.DISCRETIZATION)

            
            w1 = int(w / cnf.DISCRETIZATION)
            l1 = int(l / cnf.DISCRETIZATION)


            

            bev_corners = get_corners(x1, y1, w1, l1, yaw)
            
            bev_corners = bev_corners.astype( int )            

            
            SFA_3D_Ground_Truth_Bounding_Box_Dictionary[ "Image_Path" ].append( img_path )


            SFA_3D_Ground_Truth_Bounding_Box_Dictionary[ "Confidence_Score_Prediction" ].append( 100 )

            SFA_3D_Ground_Truth_Bounding_Box_Dictionary[ "BEV_Corner" ].append( [ bev_corners[ 0 ][ 0 ] ,
                                                                                 bev_corners[ 0 ][ 1 ] ,
                                                                                 bev_corners[ 1 ][ 0 ] ,
                                                                                 bev_corners[ 1 ][ 1 ] ,
                                                                                 bev_corners[ 2 ][ 0 ] ,
                                                                                 bev_corners[ 2 ][ 1 ] ,
                                                                                 bev_corners[ 3 ][ 0 ] ,
                                                                                 bev_corners[ 3 ][ 1 ] ] 
                                                                                 )
                                                                                 

    SFA_3D_Ground_Truth_Bounding_Box_Dataset = pd.DataFrame( SFA_3D_Ground_Truth_Bounding_Box_Dictionary )


    # Evaluate SFA 3D Hyundai Rosbag Race Model

    SFA_3D_Rosbag_Hyundai_Evaluation_Dictionary = { "SFA_3D_Rosbag_Hyundai_Model_Path" : [] ,
                                                   "SFA_3D_Rosbag_Average_Precision" : [] ,
                                                   "SFA_3D_Rosbag_Average_Precision_0.5_0.55_0.95" : [] ,
                                                   "SFA_3D_Rosbag_Hyundai_Average_IOU" : [] }
    

    for SFA_3D_Rosbag_Hyundai_Model_Path in sorted( os.listdir( NAME_FOLDER_OF_SFA3D_MODEL_ROSBAG_HYUNDAI ) ) :

        if ( any( [ i in SFA_3D_Rosbag_Hyundai_Model_Path for i in LIST_SFA_3D_MODEL_ROSBAG_HYUNDAI_EVALUATED ] ) & ( "Utils" not in SFA_3D_Rosbag_Hyundai_Model_Path  ) ) :

            print( "Checking SFA 3D Rosbag Hyundai Model " + str( SFA_3D_Rosbag_Hyundai_Model_Path ) )

            configs[ "pretrained_path" ] = NAME_FOLDER_OF_SFA3D_MODEL_ROSBAG_HYUNDAI + SFA_3D_Rosbag_Hyundai_Model_Path

            model = create_model(configs)
            print('\n\n' + '-*=' * 30 + '\n\n')
            assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
            model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
            print('Loaded weights from {}\n'.format(configs.pretrained_path))

            #configs.device = "cpu"#torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
            model = model.to(device=configs.device)

            out_cap = None

            model.eval()

            test_dataloader = create_val_dataloader(configs)

            configs.is_using_Output_Prediction = bool( configs.is_using_Output_Prediction )

            SFA_3D_Prediction_Bounding_Box_Dictionary = { "Image_Path" : [] ,
                                                   "Confidence_Score_Prediction" : [] ,
                                                   "BEV_Corner" : [] }

            with torch.no_grad():

                
                for batch_idx, batch_data in enumerate(test_dataloader):
                    metadatas, bev_maps, img_rgbs = batch_data
                    #print( metadatas )
                    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
                    t1 = time_synchronized()
                    outputs = model(input_bev_maps)
                    outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
                    outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
                    # detections size (batch_size, K, 10)
                    detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                                        outputs['dim'], K=configs.K)
                    detections = detections.cpu().numpy().astype(np.float32)
                    detections = post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)
                    t2 = time_synchronized()

                    list_of_detections = detections.copy()

                    for LiDAR_Point_Detected in range( len( list_of_detections ) ) : 



                        detections = list_of_detections[ LiDAR_Point_Detected ]  # only first batch

                        img_path = metadatas['img_path'][ LiDAR_Point_Detected ]

                        for j in range(configs.num_classes):
                            if len(detections[j]) > 0:

                                i = 0
                                for det in detections[j]:

                                    i = i + 1
                                    # (scores-0:1, x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
                                    _score, _x, _y, _z, _h, _w, _l, _yaw = det
                                    bev_corners = get_corners(_x, _y, _w, _l, _yaw)
                                    #drawRotatedBox(img_path, img, _x, _y, _w, _l, _yaw, cnf.colors[int(j)], writing_mode="Prediction" , confidence_score_prediction= _score )

                                    SFA_3D_Prediction_Bounding_Box_Dictionary[ "Image_Path" ].append( img_path )

                                    SFA_3D_Prediction_Bounding_Box_Dictionary[ "Confidence_Score_Prediction" ].append( float( _score ) )

                                    SFA_3D_Prediction_Bounding_Box_Dictionary[ "BEV_Corner" ].append( [ bev_corners[ 0 ][ 0 ] ,
                                                                                                        bev_corners[ 0 ][ 1 ] ,
                                                                                                        bev_corners[ 1 ][ 0 ] ,
                                                                                                        bev_corners[ 1 ][ 1 ] ,
                                                                                                        bev_corners[ 2 ][ 0 ] ,
                                                                                                        bev_corners[ 2 ][ 1 ] ,
                                                                                                        bev_corners[ 3 ][ 0 ] ,
                                                                                                        bev_corners[ 3 ][ 1 ] ] 
                                                                                                        )
                                    #print( "Prediction SFA 3D BEV : " + str( bev_corners ))                                
                    
            
            SFA_3D_Prediction_Bounding_Box_Dataset = pd.DataFrame( SFA_3D_Prediction_Bounding_Box_Dictionary )


            Object_Detection_Prediction_SFA_3D_Dataset_IOU_Dictionary = { "Object_Detection_Prediction_Index" : [] ,
                                                             "Confidence_Score_SFA_3D_Dataset" : [] ,
                                                             "Object_Detection_IOU" : [] }

            for Object_Detection_Prediction_SFA_3D_Dataset_Index in sorted( list( SFA_3D_Prediction_Bounding_Box_Dataset.index ) ) :

                Object_Detection_Prediction_SFA_3D_Dataset = SFA_3D_Prediction_Bounding_Box_Dataset.loc[ Object_Detection_Prediction_SFA_3D_Dataset_Index ]

                list_of_IOU_Object_Detection_SFA_3D_Dataset = []

                for _ , Object_Detection_in_Ground_Truth_SFA_3D_Dataset in SFA_3D_Ground_Truth_Bounding_Box_Dataset[ SFA_3D_Ground_Truth_Bounding_Box_Dataset[ "Image_Path"] == str( Object_Detection_Prediction_SFA_3D_Dataset[ "Image_Path" ])].iterrows() :

                    list_of_IOU_Object_Detection_SFA_3D_Dataset.append( bb_intersection_over_union( Object_Detection_Prediction_SFA_3D_Dataset[ "BEV_Corner" ] , 
                                                                                                Object_Detection_in_Ground_Truth_SFA_3D_Dataset[ "BEV_Corner" ]))
                    
                Object_Detection_Prediction_SFA_3D_Dataset_IOU_Dictionary[ "Object_Detection_Prediction_Index" ].append( Object_Detection_Prediction_SFA_3D_Dataset_Index )

                Object_Detection_Prediction_SFA_3D_Dataset_IOU_Dictionary[ "Confidence_Score_SFA_3D_Dataset" ].append( Object_Detection_Prediction_SFA_3D_Dataset[ "Confidence_Score_Prediction" ])

                Object_Detection_Prediction_SFA_3D_Dataset_IOU_Dictionary[ "Object_Detection_IOU" ].append( max( list_of_IOU_Object_Detection_SFA_3D_Dataset ))

            Object_Detection_Prediction_SFA_3D_Dataset_IOU = pd.DataFrame( Object_Detection_Prediction_SFA_3D_Dataset_IOU_Dictionary )

            #print( Object_Detection_Prediction_SFA_3D_Dataset_IOU.to_string() )

            Object_Detection_Prediction_SFA_3D_Dataset_IOU = Object_Detection_Prediction_SFA_3D_Dataset_IOU.sort_values( "Confidence_Score_SFA_3D_Dataset" , ascending=False )

            Object_Detection_Prediction_SFA_3D_Average_Precision_Dictionary = {}

            for iou_threshold in range( 50 , 100 , 5 ):

                IOU_Treshold_Minimum_Value = iou_threshold

                IOU_Treshold_Minimum_Value = IOU_Treshold_Minimum_Value / 100 

                Object_Detection_Prediction_SFA_3D_Dataset_IOU[ "SFA_3D_Dataset_Prediction_Result" ] = Object_Detection_Prediction_SFA_3D_Dataset_IOU[ "Object_Detection_IOU" ].apply( lambda row : 1 if row >= IOU_Treshold_Minimum_Value else 0 )

                y_true = np.array( Object_Detection_Prediction_SFA_3D_Dataset_IOU[ "SFA_3D_Dataset_Prediction_Result"])
                y_scores = np.array( Object_Detection_Prediction_SFA_3D_Dataset_IOU[ "Confidence_Score_SFA_3D_Dataset" ] )

                #print( "SFA 3D Dataset Ground Truth is : " + str( y_true ))

                #print( "SFA 3D Dataset Prediction is : " + str( y_scores ) )
                
                Object_Detection_Prediction_SFA_3D_Average_Precision_Dictionary[ str( IOU_Treshold_Minimum_Value ) ] = average_precision_score(y_true, y_scores)

            print( "Average Precision of Object Detection SFA 3D Dataset is : " + str( Object_Detection_Prediction_SFA_3D_Average_Precision_Dictionary ) )

            print( "Average Precision Object Detection SFA 3D with IOU Treshold Minimul 0.50 , 0.55 , .. 0.95 is : " + str( np.array( [ Object_Detection_Prediction_SFA_3D_Average_Precision_Dictionary[ i ] for i in Object_Detection_Prediction_SFA_3D_Average_Precision_Dictionary.keys() ]).mean()))

            print( "Average IOU of Object Detection SFA 3D Is : " + str( Object_Detection_Prediction_SFA_3D_Dataset_IOU[ "Object_Detection_IOU" ].mean() ) )

            SFA_3D_Rosbag_Hyundai_Evaluation_Dictionary[ "SFA_3D_Rosbag_Hyundai_Model_Path" ].append( str( SFA_3D_Rosbag_Hyundai_Model_Path ) )

            SFA_3D_Rosbag_Hyundai_Evaluation_Dictionary[ "SFA_3D_Rosbag_Average_Precision" ].append( Object_Detection_Prediction_SFA_3D_Average_Precision_Dictionary )

            SFA_3D_Rosbag_Hyundai_Evaluation_Dictionary[ "SFA_3D_Rosbag_Average_Precision_0.5_0.55_0.95" ].append( np.array( [ Object_Detection_Prediction_SFA_3D_Average_Precision_Dictionary[ i ] for i in Object_Detection_Prediction_SFA_3D_Average_Precision_Dictionary.keys() ]).mean() )
            
            SFA_3D_Rosbag_Hyundai_Evaluation_Dictionary[ "SFA_3D_Rosbag_Hyundai_Average_IOU" ].append( Object_Detection_Prediction_SFA_3D_Dataset_IOU[ "Object_Detection_IOU" ].mean() )

            print( "-------------------------------------------------------------------")


    SFA_3D_Rosbag_Hyundai_Evaluation_Dataset = pd.DataFrame( SFA_3D_Rosbag_Hyundai_Evaluation_Dictionary )

    SFA_3D_Rosbag_Hyundai_Evaluation_Dataset = SFA_3D_Rosbag_Hyundai_Evaluation_Dataset.sort_values( "SFA_3D_Rosbag_Average_Precision_0.5_0.55_0.95" , ascending= False )

    print( SFA_3D_Rosbag_Hyundai_Evaluation_Dataset.to_string() )

    export = SFA_3D_Rosbag_Hyundai_Evaluation_Dataset.to_csv( "SFA_3D_Rosbag_Hyundai_Evaluation_SFA_3D_Model_Input_800_Range_100_M.csv" , index = False )

