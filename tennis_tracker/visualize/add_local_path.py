"""Adds the path to local images if others are done on a remote machine"""

import os

import numpy as np

from tennis_tracker.player_location.extract_keypoints import (
    read_json_file,
    write_to_json_file,
)
from tennis_tracker.player_location.homography import transform_points


def get_split_box(box):
    split_box = box.split(" ")[1:]
    split_box = [float(x) for x in split_box]
    return split_box


if __name__ == "__main__":
    FILE_PATH = ""
    NEW_FILE_PATH = ""
    LOCAL_PATH_TO_TENNIS_TRACKER = ""

    data = read_json_file(FILE_PATH)
    



    # redo the transform coords
    for key in data:
        
        # redo the transform coords
        m = np.array(data[key]['m'])
        boxes = data[key]['boxes']
        new_boxes = []
        for box in boxes:
            split_box = box.split(" ")[1:]
            split_box = [float(x) for x in split_box]
            new_boxes.append(split_box)
        actual_key = key.split("/tennis_tracker/")[1]
        actual_path = os.path.join(LOCAL_PATH_TO_TENNIS_TRACKER, actual_key).replace(" &", "&")
        data[key]['actual_path'] = actual_path
        
    write_to_json_file(NEW_FILE_PATH, data)