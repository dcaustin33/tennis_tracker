import os

import cv2
import numpy as np
import torch
from groundingdino.util.inference import batch_predict, load_image_quarters, load_model
from tqdm import tqdm

from tennis_tracker.player_location.extract_keypoints import (
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
    TEXT_PROMPT = "tennis ball"
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    JSON_PATH = (
        # "/home/da2986/tennis_tracker/tennis_tracker/pseudo_label/clean_labels.json"
        "/home/da2986/tennis_tracker/tennis_tracker/pseudo_label/labels_V010_v3.json"
    )

    data = read_json_file(JSON_PATH)
    img_paths = [img_path for img_path in data.keys()]

    batch_size = 3
    OUTPUT_JSON_PATH = "/home/da2986/tennis_tracker/tennis_tracker/ball_tracking/labels_V010_v3.json"
    
    if os.path.exists(OUTPUT_JSON_PATH):
        os.remove(OUTPUT_JSON_PATH)

    for i in tqdm(range(0, len(img_paths), batch_size)):
        batch_images = img_paths[i : i + batch_size]
        loaded_images = []
        image_paths = []
        for image in batch_images:
            image_paths.extend([image] * 4)
            image_source, image = load_image_quarters(image)
            loaded_images.extend(image)
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
        final_boxes = []
        all_im_boxes = []
        all_logits = []
        previous_path = ""
        for im_num in range(len(batch_images) * 4):
            # get all the boxes that correspond to this image
            im_boxes = boxes[torch.Tensor(boxes_to_im) == im_num]
            im_logits = logits[torch.Tensor(boxes_to_im) == im_num]
            if previous_path != image_paths[im_num]:
                all_boxes = []
                all_im_boxes = []
                previous_path = image_paths[im_num]
            if len(im_boxes) > 0:
                for box, logit in zip(im_boxes, im_logits):
                    box[0] *= 0.5
                    box[1] *= 0.5
                    box[2] *= 0.5
                    box[3] *= 0.5
                    if im_num % 4 == 0:
                        pass
                    elif im_num % 4 == 1:
                        box[0] += 0.5
                    elif im_num % 4 == 2:
                        box[1] += 0.5
                    elif im_num % 4 == 3:
                        box[0] += 0.5
                        box[1] += 0.5
                    all_boxes.append(f"0 {box[0]} {box[1]} {box[2]} {box[3]}")
                    all_im_boxes.append(box)
                    all_logits.append(logit.item())
                # now we translate to the world coords
                image_dims = data[image_paths[im_num]]['image_dims'].copy()
                m = np.array(data[image_paths[im_num]]['m'].copy())
                transformed_points = transform_points(m, all_im_boxes, image_dims)
                data[image_paths[im_num]]['ball_tracking_boxes'] = all_boxes
                data[image_paths[im_num]]['ball_tracking_transformed_coords'] = transformed_points
                data[image_paths[im_num]]['ball_logits'] = all_logits
            elif im_num % 4 == 3 and len(all_im_boxes) > 0:
                data[image_paths[im_num]]['ball_tracking_boxes'] = []
                data[image_paths[im_num]]['ball_tracking_transformed_coords'] = []
                data[image_paths[im_num]]['ball_logits'] = []
            
            
    write_to_json_file(OUTPUT_JSON_PATH, data)
