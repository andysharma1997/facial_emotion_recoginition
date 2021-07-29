from src.utilities.objects import Frame
from src.utilities import sken_logger
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

logger = sken_logger.get_logger("face_proximity")


def face_proximity_detection(frame1: Frame, frame2: Frame, proximity_threshold=0.1):
    logger.info(
        "Calculating proximity for frame_id={} with frame_id={}".format(frame1.frame_number, frame2.frame_number))
    distance_matrix = euclidean_distances(frame1.get_all_face_positions(), frame2.get_all_face_positions())
    min_val, max_val = distance_matrix.min(), distance_matrix.max()
    normalized_distance_matrix = (distance_matrix - min_val) / (min_val - max_val)
    proximity_results = {}
    print(normalized_distance_matrix)
    for i in range(len(normalized_distance_matrix)):
        min_distance_idx = np.argmin(normalized_distance_matrix[i])
        if normalized_distance_matrix[i][min_distance_idx] < proximity_threshold:
            proximity_results[i] = min_distance_idx
        else:
            proximity_results[i] = None
    return proximity_results


def get_frame_features_combinations(frame1: Frame, frame2: Frame, proximity_result: dict, unit_to_combine: str):
    unit_info_1 = [face[unit_to_combine] for face in frame1.get_face_info()]
    unit_info_2 = [face[unit_to_combine] for face in frame2.get_face_info()]
    for key, val in proximity_result.items():
        if val is not None:
            for k1 in unit_info_1[key].keys():
                if k1 in unit_info_2[val].keys():
                    unit_info_2[val][k1] = float(unit_info_2[val][k1])+ float(unit_info_1[key][k1])
                else:
                    unit_info_2[val][k1] = unit_info_1[key][k1]
        else:
            unit_info_2.append(unit_info_1[key])
    return unit_info_2


def get_frame_box_combinations(frame1: Frame, frame2: Frame, proximity_result: dict):
    f1 = [face['face_box'] for face in frame1.get_face_info()]
    f2 = [face['face_box'] for face in frame2.get_face_info()]
    for key, val in proximity_result.items():
        if val is None:
            f2.append(f1[key])
    return f2