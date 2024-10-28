import os

import cv2
import numpy as np
import torch
from groundingdino.util.inference import batch_predict, load_image, load_model
from tqdm import tqdm

from tennis_tracker.download_data.extract_keypoints import (
    read_json_file,
    write_to_json_file,
)
from tennis_tracker.player_location.homography import transform_points


def output_point(m: np.array, points: np.array) -> list:
    """points should be in shape (-1, 1, 2)"""
    outputs = cv2.perspectiveTransform(points, m)
    # output will be -1, 1, 2
    return outputs.reshape(-1, 2)

def cut_image_into_quarters(image: np.array) -> list:
    height = (image.shape[1] // 2) * 2
    width = (image.shape[2] // 2) * 2
    image = image[:, :height, :width]
    return [image[:, :height//2, :width//2], image[:, :height//2, width//2:], image[:, height//2:, :width//2], image[:, height//2:, width//2:]]

if __name__ == "__main__":

    model = load_model(
        "/home/da2986/tennis_tracker/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "/home/da2986/tennis_tracker/GroundingDINO/groundingdino_swint_ogc.pth",
    )
    # model = torch.compile(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    TEXT_PROMPT = "yellow tennis ball"
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    JSON_PATH = (
        "/home/da2986/tennis_tracker/tennis_tracker/psudeo_label/clean_labels.json"
    )

    data = read_json_file(JSON_PATH)
    img_paths = [img_path for img_path in data.keys()]

    batch_size = 10
    OUTPUT_JSON_PATH = "/home/da2986/tennis_tracker/tennis_tracker/ball_tracking/labels.json"
    
    if os.path.exists(OUTPUT_JSON_PATH):
        os.remove(OUTPUT_JSON_PATH)

    for i in tqdm(range(0, len(img_paths), batch_size)):
        batch_images = img_paths[i : i + batch_size]
        loaded_images = []
        for image in batch_images:
            image_source, image = load_image(image)
            quarters = cut_image_into_quarters(image)
            loaded_images.extend(quarters)
        input_images = torch.stack(loaded_images)
        boxes, logits, boxes_to_im = batch_predict(
            model=model,
            preprocessed_images=input_images,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device=device,
        )
        lines = []
        for im_num in range(len(batch_images)):
            # get all the boxes that correspond to this image
            im_boxes = boxes[torch.Tensor(boxes_to_im) == im_num]
            all_boxes = []
            for box in im_boxes:
                all_boxes.append(f"0 {box[0]} {box[1]} {box[2]} {box[3]}")
            data[batch_images[im_num]]['ball_tracking_boxes'] = all_boxes
            lines.append(all_boxes)
            
            # now we translate to the world coords
            image_dims = data[batch_images[im_num]]['image_dims'].copy()
            m = np.array(data[batch_images[im_num]]['m'].copy())
            transformed_points = transform_points(m, im_boxes, image_dims)
            data[batch_images[im_num]]['ball_tracking_transformed_coords'] = transformed_points
            
            
    write_to_json_file(OUTPUT_JSON_PATH, data)
