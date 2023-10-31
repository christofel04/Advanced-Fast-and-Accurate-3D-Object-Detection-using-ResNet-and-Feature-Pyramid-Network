import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from shapely.geometry import Polygon

def bb_intersection_over_union(boxA, boxB):
    polygon = Polygon([(boxA[ 0 ], boxA[ 1 ]), (boxA[ 2 ], boxA[ 3 ]), (boxA[ 4 ], boxA[ 5 ]), (boxA[ 6 ], boxA[ 7 ] ) ] )
    other_polygon = Polygon([(boxB[ 0 ], boxB[ 1 ]), (boxB[ 2 ], boxB[ 3 ]), (boxB[ 4 ], boxB[ 5 ]), ( boxB[ 6 ], boxB[ 7 ])])
    if (polygon.intersects(other_polygon)):
        intersection = polygon.intersection(other_polygon)
        union_area = polygon.area + other_polygon.area - intersection.area
        iou = intersection.area / union_area
        return iou
    return 0


def get_results(file_path):
    results = { "img_path" : [] , "confidence" : [] , "coordinate" : [] }
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            object_detection = line.strip().split()
            img_path = object_detection[0].split("/")[-1]
            confidence = float(object_detection[1])
            coordinates = [float(coord) for coord in object_detection[2:]]
            results["img_path"].append(img_path)
            results["confidence"].append(confidence)
            results["coordinate"].append(coordinates)
        dataframe = pd.DataFrame(results)

        return dataframe
    

iou_data = {"index":[],"confidence":[],"iou":[]}
AP_dict = {}

gt_file = "./data_process/Output_Ground_TruthGround_Truth_Dataset_SFA_3D_from_Bag.txt"
pred_file = "Output_PredictionGround_Truth_Dataset_SFA_3D_from_Bag.txt"

gt_results = get_results(gt_file)
pred_results = get_results(pred_file)

for idx in sorted(pred_results.index):
    pred_data = pred_results.loc[idx]
    filtered = gt_results[gt_results["img_path"] == str(pred_data["img_path"])]
    iou_list = [bb_intersection_over_union(pred_data["coordinate"], gt_data["coordinate"])
    for _, gt_data in filtered.iterrows()]

    iou_data["index"].append(idx)
    iou_data["confidence"].append(pred_data["confidence"])
    iou_data["iou"].append(max(iou_list))

sorted_iou = pd.DataFrame(iou_data).sort_values("confidence", ascending=True)

for iou_threshold in range(50,100,5):
    iou_threshold = iou_threshold / 100 
    sorted_iou["result"] = sorted_iou["iou"].apply(lambda row : 1 if row >= iou_threshold else 0)
    y_true = np.array(sorted_iou["result"])
    y_scores = np.array(sorted_iou["confidence" ])
    AP_dict[str( iou_threshold )] = average_precision_score(y_true, y_scores)

print( "Average Precision: " + str( AP_dict ) )

print( "Average Precision Incremental: " + str( np.array( [ AP_dict[ i ] for i in AP_dict.keys() ]).mean()))

print( "Average IOU of Object Detection SFA 3D Is : " + str(sorted_iou["iou"].mean()))

print( "-------------------------------------------------------------------")