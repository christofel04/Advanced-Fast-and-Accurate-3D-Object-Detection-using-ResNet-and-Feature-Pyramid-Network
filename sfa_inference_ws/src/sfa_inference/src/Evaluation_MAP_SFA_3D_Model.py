from sklearn.metrics import average_precision_score
from shapely.geometry import Polygon

def bb_intersection_over_union(boxA, boxB):
    

    polygon = Polygon([(boxA[ 0 ], boxA[ 1 ]), (boxA[ 2 ], boxA[ 3 ]), (boxA[ 4 ], boxA[ 5 ]), (boxA[ 6 ], boxA[ 7 ] ) ] )
    other_polygon = Polygon([(boxB[ 0 ], boxB[ 1 ]), (boxB[ 2 ], boxB[ 3 ]), (boxB[ 4 ], boxB[ 5 ]), ( boxB[ 6 ], boxB[ 7 ])])
    intersection = polygon.intersection(other_polygon)
    print(intersection.area)

    Area_of_Union_2_Bounding_Box = polygon.area + other_polygon.area - intersection.area

    iou = intersection.area / Area_of_Union_2_Bounding_Box

    # return the intersection over union value
    return iou

import pandas as pd

import numpy as np

FILE_OF_GROUND_TRUTH_SFA_3D_DATASET = "./data_process/Output_Ground_TruthGround_Truth_Dataset_SFA_3D_from_Bag.txt"

File_of_Ground_Truth_SFA_3D_Dataset_Dictionary = { "Image_Path_Dataset_SFA_3D" : [] ,
                                                  "Confidence_Score_SFA_3D_Dataset" : [] ,
                                                  "Coordinate_of_SFA_3D_Object_Dataset" : [] }

import os

"""

if os.path.exists( FILE_OF_GROUND_TRUTH_SFA_3D_DATASET ):

    os.system( "rm " + str( FILE_OF_GROUND_TRUTH_SFA_3D_DATASET ))

    os.system( "touch " + str( FILE_OF_GROUND_TRUTH_SFA_3D_DATASET ) )

"""

for File_of_Ground_Truth_SFA_3D_Dataset_Data_per_Object_Detection in open( FILE_OF_GROUND_TRUTH_SFA_3D_DATASET , "r" ).readlines() :

    Object_Detection_of_Ground_Truth_SFA_3D_Dataset = File_of_Ground_Truth_SFA_3D_Dataset_Data_per_Object_Detection.replace( "/n" , "" ).split( " " )

    File_of_Ground_Truth_SFA_3D_Dataset_Dictionary[ "Image_Path_Dataset_SFA_3D"].append( Object_Detection_of_Ground_Truth_SFA_3D_Dataset[ 0 ].split( "/" )[ - 1 ] )

    File_of_Ground_Truth_SFA_3D_Dataset_Dictionary[ "Confidence_Score_SFA_3D_Dataset" ].append( float( Object_Detection_of_Ground_Truth_SFA_3D_Dataset[ 1 ] ) )

    File_of_Ground_Truth_SFA_3D_Dataset_Dictionary[ "Coordinate_of_SFA_3D_Object_Dataset" ].append( [ float( Object_Detection_of_Ground_Truth_SFA_3D_Dataset[ 2 ]) ,
                                                                                                      float( Object_Detection_of_Ground_Truth_SFA_3D_Dataset[ 3 ]) ,
                                                                                                      float( Object_Detection_of_Ground_Truth_SFA_3D_Dataset[ 4 ]) ,
                                                                                                      float( Object_Detection_of_Ground_Truth_SFA_3D_Dataset[ 5 ]) ,
                                                                                                      float( Object_Detection_of_Ground_Truth_SFA_3D_Dataset[ 6 ]) ,
                                                                                                     float( Object_Detection_of_Ground_Truth_SFA_3D_Dataset[ 7 ]) ,
                                                                                                      float( Object_Detection_of_Ground_Truth_SFA_3D_Dataset[ 8 ]) ,
                                                                                                       float( Object_Detection_of_Ground_Truth_SFA_3D_Dataset[ 9 ]) ])

File_of_Ground_Truth_SFA_3D_Dataset = pd.DataFrame( File_of_Ground_Truth_SFA_3D_Dataset_Dictionary )

print( File_of_Ground_Truth_SFA_3D_Dataset.to_string())


FILE_OF_PREDICTION_SFA_3D_DATASET = "Output_PredictionGround_Truth_Dataset_SFA_3D_from_Bag.txt"

File_of_Ground_Truth_SFA_3D_Dataset_Dictionary = { "Image_Path_Dataset_SFA_3D" : [] ,
                                                  "Confidence_Score_SFA_3D_Dataset" : [] ,
                                                  "Coordinate_of_SFA_3D_Object_Dataset" : [] }

"""
if os.path.exists( FILE_OF_PREDICTION_SFA_3D_DATASET ):

    os.system( "rm " + str( FILE_OF_PREDICTION_SFA_3D_DATASET ))

    os.system( "touch " + str( FILE_OF_PREDICTION_SFA_3D_DATASET ) )
"""

for File_of_Ground_Truth_SFA_3D_Dataset_Data_per_Object_Detection in open( FILE_OF_PREDICTION_SFA_3D_DATASET , "r" ).readlines() :

    Object_Detection_of_Ground_Truth_SFA_3D_Dataset = File_of_Ground_Truth_SFA_3D_Dataset_Data_per_Object_Detection.replace( "/n" , "" ).split( " " )

    File_of_Ground_Truth_SFA_3D_Dataset_Dictionary[ "Image_Path_Dataset_SFA_3D"].append( Object_Detection_of_Ground_Truth_SFA_3D_Dataset[ 0 ].split( "/" )[ - 1 ] )

    File_of_Ground_Truth_SFA_3D_Dataset_Dictionary[ "Confidence_Score_SFA_3D_Dataset" ].append( float( Object_Detection_of_Ground_Truth_SFA_3D_Dataset[ 1 ] ) )

    File_of_Ground_Truth_SFA_3D_Dataset_Dictionary[ "Coordinate_of_SFA_3D_Object_Dataset" ].append( [ float( Object_Detection_of_Ground_Truth_SFA_3D_Dataset[ 2 ]) ,
                                                                                                      float( Object_Detection_of_Ground_Truth_SFA_3D_Dataset[ 3 ]) ,
                                                                                                      float( Object_Detection_of_Ground_Truth_SFA_3D_Dataset[ 4 ]) ,
                                                                                                      float( Object_Detection_of_Ground_Truth_SFA_3D_Dataset[ 5 ]) ,
                                                                                                    float( Object_Detection_of_Ground_Truth_SFA_3D_Dataset[ 6 ]) ,
                                                                                                     float( Object_Detection_of_Ground_Truth_SFA_3D_Dataset[ 7 ]) ,
                                                                                                      float( Object_Detection_of_Ground_Truth_SFA_3D_Dataset[ 8 ]) ,
                                                                                                       float( Object_Detection_of_Ground_Truth_SFA_3D_Dataset[ 9 ]) ])

File_of_Prediction_SFA_3D_Dataset = pd.DataFrame( File_of_Ground_Truth_SFA_3D_Dataset_Dictionary )

Object_Detection_Prediction_SFA_3D_Dataset_IOU_Dictionary = { "Object_Detection_Prediction_Index" : [] ,
                                                             "Confidence_Score_SFA_3D_Dataset" : [] ,
                                                             "Object_Detection_IOU" : [] }

for Object_Detection_Prediction_SFA_3D_Dataset_Index in sorted( list( File_of_Prediction_SFA_3D_Dataset.index ) ) :

    Object_Detection_Prediction_SFA_3D_Dataset = File_of_Prediction_SFA_3D_Dataset.loc[ Object_Detection_Prediction_SFA_3D_Dataset_Index ]

    list_of_IOU_Object_Detection_SFA_3D_Dataset = []

    for _ , Object_Detection_in_Ground_Truth_SFA_3D_Dataset in File_of_Ground_Truth_SFA_3D_Dataset[ File_of_Ground_Truth_SFA_3D_Dataset[ "Image_Path_Dataset_SFA_3D"] == str( Object_Detection_Prediction_SFA_3D_Dataset[ "Image_Path_Dataset_SFA_3D" ])].iterrows() :

        list_of_IOU_Object_Detection_SFA_3D_Dataset.append( bb_intersection_over_union( Object_Detection_Prediction_SFA_3D_Dataset[ "Coordinate_of_SFA_3D_Object_Dataset" ] , 
                                                                                       Object_Detection_in_Ground_Truth_SFA_3D_Dataset[ "Coordinate_of_SFA_3D_Object_Dataset" ]))
        
    Object_Detection_Prediction_SFA_3D_Dataset_IOU_Dictionary[ "Object_Detection_Prediction_Index" ].append( Object_Detection_Prediction_SFA_3D_Dataset_Index )

    Object_Detection_Prediction_SFA_3D_Dataset_IOU_Dictionary[ "Confidence_Score_SFA_3D_Dataset" ].append( Object_Detection_Prediction_SFA_3D_Dataset[ "Confidence_Score_SFA_3D_Dataset" ])

    Object_Detection_Prediction_SFA_3D_Dataset_IOU_Dictionary[ "Object_Detection_IOU" ].append( max( list_of_IOU_Object_Detection_SFA_3D_Dataset ))

Object_Detection_Prediction_SFA_3D_Dataset_IOU = pd.DataFrame( Object_Detection_Prediction_SFA_3D_Dataset_IOU_Dictionary )

print( Object_Detection_Prediction_SFA_3D_Dataset_IOU.to_string() )

Object_Detection_Prediction_SFA_3D_Dataset_IOU = Object_Detection_Prediction_SFA_3D_Dataset_IOU.sort_values( "Confidence_Score_SFA_3D_Dataset" , ascending=True )

Object_Detection_Prediction_SFA_3D_Average_Precision_Dictionary = {}

for iou_threshold in range( 50 , 100 , 5 ):

    IOU_Treshold_Minimum_Value = IOU_Treshold_Minimum_Value / 100 

    Object_Detection_Prediction_SFA_3D_Dataset_IOU[ "SFA_3D_Dataset_Prediction_Result" ] = Object_Detection_Prediction_SFA_3D_Dataset_IOU[ "Object_Detection_IOU" ].apply( lambda row : 1 if row >= IOU_Treshold_Minimum_Value else 0 )

    y_true = np.array( Object_Detection_Prediction_SFA_3D_Dataset_IOU[ "SFA_3D_Dataset_Prediction_Result"])
    y_scores = np.array( Object_Detection_Prediction_SFA_3D_Dataset_IOU[ "Confidence_Score_SFA_3D_Dataset" ] )

    print( "SFA 3D Dataset Ground Truth is : " + str( y_true ))

    print( "SFA 3D Dataset Prediction is : " + str( y_scores ) )
    
    Object_Detection_Prediction_SFA_3D_Average_Precision_Dictionary[ str( IOU_Treshold_Minimum_Value ) ] = average_precision_score(y_true, y_scores)

print( "Average Precision of Object Detection SFA 3D Dataset is : " + str( Object_Detection_Prediction_SFA_3D_Average_Precision_Dictionary ) )

print( "Average Precision Object Detection SFA 3D with IOU Treshold Minimul 0.50 , 0.55 , .. 0.95 is : " + str( np.array( [ Object_Detection_Prediction_SFA_3D_Average_Precision_Dictionary[ i ] for i in Object_Detection_Prediction_SFA_3D_Average_Precision_Dictionary.keys() ]).mean()))

print( "Average IOU of Object Detection SFA 3D Is : " + str( Object_Detection_Prediction_SFA_3D_Dataset_IOU[ "Object_Detection_IOU" ].mean() ) )

print( "-------------------------------------------------------------------")