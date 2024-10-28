import os

import numpy as np

from tennis_tracker.download_data.extract_keypoints import (
    read_json_file,
    write_to_json_file,
)
from tennis_tracker.player_location.homography import transform_points

if __name__ == "__main__":
    FILE_PATH = "/Users/derek/Desktop/tennis_tracker/tennis_tracker/psudeo_label/labels.json"
    NEW_FILE_PATH = "/Users/derek/Desktop/tennis_tracker/tennis_tracker/psudeo_label/clean_labels.json"

    data = read_json_file(FILE_PATH)
    



    # redo the transform coords
    for key in data:
        # swap the image dims
        data[key]['image_dims'] = data[key]['image_dims'][::-1]
        
        # redo the transform coords
        m = np.array(data[key]['m'])
        boxes = data[key]['boxes']
        new_boxes = []
        for box in boxes:
            split_box = box.split(" ")[1:]
            split_box = [float(x) for x in split_box]
            new_boxes.append(split_box)
        data[key]['transformed_coords'] = transform_points(m, new_boxes, data[key]['image_dims'])
        actual_key = key.split("/tennis_tracker/")[1]
        actual_path = os.path.join("/Users/derek/Desktop/tennis_tracker", actual_key).replace(" &", "&")
        data[key]['actual_path'] = actual_path
        
    write_to_json_file(NEW_FILE_PATH, data)