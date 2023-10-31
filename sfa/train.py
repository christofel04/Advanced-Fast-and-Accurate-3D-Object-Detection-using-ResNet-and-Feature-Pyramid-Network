"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Goenawan Christofel Rio
# DoC: 2023.03.17
# AI & Analytics Consultant at Hyundai Company
# AI & Robotics Researcher at Korea Advanced Institute of Science and Technology
# email: christofel.goenawan@kaist.ac.kr
-----------------------------------------------------------------------------------
# Description: This script for training

"""

import time
import numpy as np
import sys
import random
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from tqdm import tqdm

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_process.kitti_dataloader import create_train_dataloader, create_val_dataloader
from models.model_utils import create_model, make_data_parallel, get_num_parameters
from utils.train_utils import create_optimizer, create_lr_scheduler, get_saved_state, save_checkpoint
from utils.torch_utils import reduce_tensor, to_python_float
from utils.misc import AverageMeter, ProgressMeter
from utils.logger import Logger
from config.train_config import parse_train_configs
from losses.losses import Compute_Loss

# Package for evaluating MAP SFA 3D Rosbag Hyundai Model

import data_process.kitti_dataset as kitti_dataset

from utils.misc import make_folder, time_synchronized
from utils.evaluation_utils import decode, post_processing, draw_predictions, convert_det_to_real_values
from utils.torch_utils import _sigmoid
import config.kitti_config as cnf

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


def main():
    configs = parse_train_configs()

    # Re-produce results
    if configs.seed is not None:
        random.seed(configs.seed)
        np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if configs.gpu_idx is not None:
        print('You have chosen a specific GPU. This will completely disable data parallelism.')

    if configs.dist_url == "env://" and configs.world_size == -1:
        configs.world_size = int(os.environ["WORLD_SIZE"])

    configs.distributed = configs.world_size > 1 or configs.multiprocessing_distributed

    if configs.multiprocessing_distributed:
        configs.world_size = configs.ngpus_per_node * configs.world_size
        mp.spawn(main_worker, nprocs=configs.ngpus_per_node, args=(configs,))
    else:
        main_worker(configs.gpu_idx, configs)


def main_worker(gpu_idx, configs):
    configs.gpu_idx = gpu_idx
    configs.device = torch.device('cpu' if configs.gpu_idx is None else 'cuda:{}'.format(configs.gpu_idx))

    if configs.distributed:
        if configs.dist_url == "env://" and configs.rank == -1:
            configs.rank = int(os.environ["RANK"])
        if configs.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            configs.rank = configs.rank * configs.ngpus_per_node + gpu_idx

        dist.init_process_group(backend=configs.dist_backend, init_method=configs.dist_url,
                                world_size=configs.world_size, rank=configs.rank)
        configs.subdivisions = int(64 / configs.batch_size / configs.ngpus_per_node)
    else:
        configs.subdivisions = int(64 / configs.batch_size)

    configs.is_master_node = (not configs.distributed) or (
            configs.distributed and (configs.rank % configs.ngpus_per_node == 0))

    if configs.is_master_node:
        logger = Logger(configs.logs_dir, configs.saved_fn)
        logger.info('>>> Created a new logger')
        logger.info('>>> configs: {}'.format(configs))
        tb_writer = SummaryWriter(log_dir=os.path.join(configs.logs_dir, 'tensorboard'))
    else:
        logger = None
        tb_writer = None

    # model
    model = create_model(configs)

    # load weight from a checkpoint
    if configs.pretrained_path is not None:
        assert os.path.isfile(configs.pretrained_path), "=> no checkpoint found at '{}'".format(configs.pretrained_path)
        model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
        if logger is not None:
            logger.info('loaded pretrained model at {}'.format(configs.pretrained_path))

    # resume weights of model from a checkpoint
    if configs.resume_path is not None:
        assert os.path.isfile(configs.resume_path), "=> no checkpoint found at '{}'".format(configs.resume_path)
        model.load_state_dict(torch.load(configs.resume_path, map_location='cpu'))
        if logger is not None:
            logger.info('resume training model from checkpoint {}'.format(configs.resume_path))

    # Data Parallel
    model = make_data_parallel(model, configs)

    # Make sure to create optimizer after moving the model to cuda
    optimizer = create_optimizer(configs, model)
    lr_scheduler = create_lr_scheduler(optimizer, configs)
    configs.step_lr_in_epoch = False if configs.lr_type in ['multi_step', 'cosin', 'one_cycle'] else True

    # resume optimizer, lr_scheduler from a checkpoint
    if configs.resume_path is not None:
        utils_path = configs.resume_path.replace('Model_', 'Utils_')
        assert os.path.isfile(utils_path), "=> no checkpoint found at '{}'".format(utils_path)
        utils_state_dict = torch.load(utils_path, map_location='cuda:{}'.format(configs.gpu_idx))
        optimizer.load_state_dict(utils_state_dict['optimizer'])
        lr_scheduler.load_state_dict(utils_state_dict['lr_scheduler'])
        configs.start_epoch = utils_state_dict['epoch'] + 1

    if configs.is_master_node:
        num_parameters = get_num_parameters(model)
        logger.info('number of trained parameters of the model: {}'.format(num_parameters))

    if logger is not None:
        logger.info(">>> Loading dataset & getting dataloader...")
    # Create dataloader
    train_dataloader, train_sampler = create_train_dataloader(configs)
    if logger is not None:
        logger.info('number of batches in training set: {}'.format(len(train_dataloader)))

    if configs.evaluate:
        val_dataloader = create_val_dataloader(configs)
        val_loss = validate(val_dataloader, model, configs)
        print('val_loss: {:.4e}'.format(val_loss))
        return
    
    Rosbag_Hyundai_Dataset_Ground_Truth_DF = None
    
    # Create Ground Truth SFA 3D Model Rosbag Hyundai

    lidar_aug = None

    dataset = kitti_dataset.KittiDataset(configs, mode='val', lidar_aug=lidar_aug, hflip_prob=0., num_samples=configs.num_samples)


    # Make ground truth Rosbag Hyundai Dataset
    if Rosbag_Hyundai_Dataset_Ground_Truth_DF == None :
            
        SFA_3D_Ground_Truth_Bounding_Box_Dictionary = { "Image_Path" : [] ,
                                        "Confidence_Score_Prediction" : [] ,
                                        "BEV_Corner" : [] }

        print( "Creating Ground Truth SFA 3D Rosbag Hyundai Dataset" )

        #configs.no_cuda= True
        
        #configs.device = "cpu"    

        for idx in range(len(dataset)):
            bev_map, labels, img_rgb, img_path = dataset.draw_img_with_label(idx)
            #print( 'Making Dataset SFA 3D Labelling Rosbag Data : ' + str( idx ))
            #calib = Calibration(img_path.replace(".png", ".txt").replace("image_2", "calib"))
            

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
                                                                                    

        Rosbag_Hyundai_Dataset_Ground_Truth_DF = pd.DataFrame( SFA_3D_Ground_Truth_Bounding_Box_Dictionary )
            
    for epoch in range(configs.start_epoch, configs.num_epochs + 1):
        if logger is not None:
            logger.info('{}'.format('*-' * 40))
            logger.info('{} {}/{} {}'.format('=' * 35, epoch, configs.num_epochs, '=' * 35))
            logger.info('{}'.format('*-' * 40))
            logger.info('>>> Epoch: [{}/{}]'.format(epoch, configs.num_epochs))

        if configs.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        loss = train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch, configs, logger, tb_writer)
        if (not configs.no_val) and (epoch % configs.checkpoint_freq == 0):
            val_dataloader = create_val_dataloader(configs)
            print('number of batches in val_dataloader: {}'.format(len(val_dataloader)))

            
            
            
            val_loss = validate(val_dataloader, model, configs , number_epoch= epoch , ground_truth_dataset = Rosbag_Hyundai_Dataset_Ground_Truth_DF )
            print('val_loss: {:.4e}'.format(val_loss))
            if tb_writer is not None:
                tb_writer.add_scalar('Val_loss', val_loss, epoch)

        # Save checkpoint
        if configs.is_master_node and ((epoch % configs.checkpoint_freq) == 0):
            model_state_dict, utils_state_dict = get_saved_state(model, optimizer, lr_scheduler, epoch, configs)
            save_checkpoint(configs.checkpoints_dir, configs.saved_fn, model_state_dict, utils_state_dict, epoch, loss)

        if not configs.step_lr_in_epoch:
            lr_scheduler.step()
            if tb_writer is not None:
                tb_writer.add_scalar('LR', lr_scheduler.get_lr()[0], epoch)

    if tb_writer is not None:
        tb_writer.close()
    if configs.distributed:
        cleanup()


def cleanup():
    dist.destroy_process_group()


def train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch, configs, logger, tb_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(len(train_dataloader), [batch_time, data_time, losses],
                             prefix="Train - Epoch: [{}/{}]".format(epoch, configs.num_epochs))

    criterion = Compute_Loss(device=configs.device)
    num_iters_per_epoch = len(train_dataloader)
    # switch to train mode
    model.train()
    start_time = time.time()
    for batch_idx, batch_data in enumerate(tqdm(train_dataloader)):
        data_time.update(time.time() - start_time)
        metadatas, imgs, targets = batch_data
        batch_size = imgs.size(0)
        global_step = num_iters_per_epoch * (epoch - 1) + batch_idx + 1
        for k in targets.keys():
            targets[k] = targets[k].to(configs.device, non_blocking=True)
        imgs = imgs.to(configs.device, non_blocking=True).float()
        outputs = model(imgs)
        total_loss, loss_stats = criterion(outputs, targets)
        # For torch.nn.DataParallel case
        if (not configs.distributed) and (configs.gpu_idx is None):
            total_loss = torch.mean(total_loss)

        # compute gradient and perform backpropagation
        total_loss.backward()
        if global_step % configs.subdivisions == 0:
            optimizer.step()
            # zero the parameter gradients
            optimizer.zero_grad()
            # Adjust learning rate
            if configs.step_lr_in_epoch:
                lr_scheduler.step()
                if tb_writer is not None:
                    tb_writer.add_scalar('LR', lr_scheduler.get_lr()[0], global_step)

        if configs.distributed:
            reduced_loss = reduce_tensor(total_loss.data, configs.world_size)
        else:
            reduced_loss = total_loss.data
        losses.update(to_python_float(reduced_loss), batch_size)
        # measure elapsed time
        # torch.cuda.synchronize()
        batch_time.update(time.time() - start_time)

        if tb_writer is not None:
            if (global_step % configs.tensorboard_freq) == 0:
                loss_stats['avg_loss'] = losses.avg
                tb_writer.add_scalars('Train', loss_stats, global_step)
        # Log message
        if logger is not None:
            if (global_step % configs.print_freq) == 0:
                logger.info(progress.get_message(batch_idx))

        start_time = time.time()
    return reduced_loss


def validate(val_dataloader, model, configs , number_epoch = 0 , ground_truth_dataset = None ):
    losses = AverageMeter('Loss', ':.4e')
    criterion = Compute_Loss(device=configs.device)
    # switch to train mode
    model.eval()

    SFA_3D_Prediction_Bounding_Box_Dictionary = { "Image_Path" : [] ,
                                                   "Confidence_Score_Prediction" : [] ,
                                                   "BEV_Corner" : [] }

    SFA_3D_Rosbag_Hyundai_Evaluation_Dictionary = { "SFA_3D_Rosbag_Average_Precision" : [] ,
                                                   "SFA_3D_Rosbag_Average_Precision_0.5_0.55_0.95" : [] ,
                                                   "SFA_3D_Rosbag_Hyundai_Average_IOU" : [] }
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(val_dataloader)):
            metadatas, imgs, targets = batch_data
            batch_size = imgs.size(0)
            for k in targets.keys():
                targets[k] = targets[k].to(configs.device, non_blocking=True)
            imgs = imgs.to(configs.device, non_blocking=True).float()
            outputs = model(imgs)
            total_loss, loss_stats = criterion(outputs, targets)
            # For torch.nn.DataParallel case
            if (not configs.distributed) and (configs.gpu_idx is None):
                total_loss = torch.mean(total_loss)

            if configs.distributed:
                reduced_loss = reduce_tensor(total_loss.data, configs.world_size)
            else:
                reduced_loss = total_loss.data
            losses.update(to_python_float(reduced_loss), batch_size)

            if ground_truth_dataset != None :
                # Evaluate MAP SFA 3D Model trains on all Rosbag Hyundai Dataset

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
                
    try :
        if ground_truth_dataset != None :
            # Evaluating MAP of SFA 3D Model trains on All Rosbag Hyundai Dataset
            
            SFA_3D_Prediction_Bounding_Box_Dataset = pd.DataFrame( SFA_3D_Prediction_Bounding_Box_Dictionary )

            SFA_3D_Ground_Truth_Bounding_Box_Dataset = ground_truth_dataset 

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

            #SFA_3D_Rosbag_Hyundai_Evaluation_Dictionary[ "SFA_3D_Rosbag_Hyundai_Model_Path" ].append( str( SFA_3D_Rosbag_Hyundai_Model_Path ) )

            SFA_3D_Rosbag_Hyundai_Evaluation_Dictionary[ "SFA_3D_Rosbag_Average_Precision" ].append( Object_Detection_Prediction_SFA_3D_Average_Precision_Dictionary )

            SFA_3D_Rosbag_Hyundai_Evaluation_Dictionary[ "SFA_3D_Rosbag_Average_Precision_0.5_0.55_0.95" ].append( np.array( [ Object_Detection_Prediction_SFA_3D_Average_Precision_Dictionary[ i ] for i in Object_Detection_Prediction_SFA_3D_Average_Precision_Dictionary.keys() ]).mean() )
            
            SFA_3D_Rosbag_Hyundai_Evaluation_Dictionary[ "SFA_3D_Rosbag_Hyundai_Average_IOU" ].append( Object_Detection_Prediction_SFA_3D_Dataset_IOU[ "Object_Detection_IOU" ].mean() )

            print( "-------------------------------------------------------------------")

            SFA_3D_Rosbag_Hyundai_Evaluation_Dataset = pd.DataFrame( SFA_3D_Rosbag_Hyundai_Evaluation_Dictionary )

            SFA_3D_Rosbag_Hyundai_Evaluation_Dataset = SFA_3D_Rosbag_Hyundai_Evaluation_Dataset.sort_values( "SFA_3D_Rosbag_Average_Precision_0.5_0.55_0.95" , ascending= False )

            export = SFA_3D_Rosbag_Hyundai_Evaluation_Dataset.to_csv( os.path.join( configs.checkpoints_dir , "SFA_3D_Rosbag_Hyundai_Evaluation_SFA_3D_Model_{}_Epoch_{}.csv".format( configs.saved_fn , number_epoch ) ) , index = False )

    except Exception as e :

        print( "Cant make evaluation SFA 3D Model on Rosbag Hyundai Dataset...." )

        print( "Error is : " + str( e ))

        print( "----------------------------------------------------------" )


    return losses.avg


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        try:
            cleanup()
            sys.exit(0)
        except SystemExit:
            os._exit(0)
