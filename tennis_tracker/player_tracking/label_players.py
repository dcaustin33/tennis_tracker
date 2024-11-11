import argparse
import os

import cv2
import numpy as np
import torch
from groundingdino.util.inference import batch_predict, load_image, load_model
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="../labels/labels.json")
    parser.add_argument("--output_json_path", type=str, default="../labels/labels.json")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_config_path", type=str, default="../../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--model_checkpoint_path", type=str, default="../../GroundingDINO/groundingdino_swint_ogc.pth")
    parser.add_argument("--text_prompt", type=str, default="tennis player")
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()

    DEVICE = args.device
    TEXT_PROMPT = args.text_prompt
    BOX_THRESHOLD = args.box_threshold
    TEXT_THRESHOLD = args.text_threshold
    JSON_PATH = args.json_path

    model = load_model(
        args.model_config_path,
        args.model_checkpoint_path,
    )
    model = model.to(DEVICE)
    data = read_json_file(JSON_PATH)
    img_paths = [img_path for img_path in data.keys()]

    batch_size = args.batch_size
    OUTPUT_JSON_PATH = args.output_json_path
    
    if os.path.exists(OUTPUT_JSON_PATH):
        os.remove(OUTPUT_JSON_PATH)

    for i in tqdm(range(0, len(img_paths), batch_size)):
        batch_images = img_paths[i : i + batch_size]
        loaded_images = []
        for image in batch_images:
            image_source, image = load_image(image)
            loaded_images.append(image)
        input_images = torch.stack(loaded_images)
        boxes, logits, boxes_to_im = batch_predict(
            model=model,
            preprocessed_images=input_images,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=DEVICE,
        )
        lines = []
        for im_num in range(len(batch_images)):
            # get all the boxes that correspond to this image
            im_boxes = boxes[torch.Tensor(boxes_to_im) == im_num]
            current_logits = logits[torch.Tensor(boxes_to_im) == im_num]
            # sort the boxes by logits
            sorted_boxes = im_boxes[current_logits.argsort(descending=True)]
            all_boxes = []
            for box in sorted_boxes[:3]:
                all_boxes.append(f"0 {box[0]} {box[1]} {box[2]} {box[3]}")
            data[batch_images[im_num]]['boxes'] = all_boxes
            lines.append(all_boxes)
            
            # now we translate to the world coords
            image_dims = data[batch_images[im_num]]['image_dims'].copy()
            m = np.array(data[batch_images[im_num]]['m'].copy())
            transformed_points = transform_points(m, sorted_boxes[:3], image_dims)
            data[batch_images[im_num]]['transformed_coords'] = transformed_points
            
            
    write_to_json_file(OUTPUT_JSON_PATH, data)
