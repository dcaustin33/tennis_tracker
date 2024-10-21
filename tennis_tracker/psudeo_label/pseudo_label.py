import os

import cv2
import torch
from groundingdino.util.inference import (
    annotate,
    batch_predict,
    load_image,
    load_model,
    predict,
)

from tennis_tracker.download_data.extract_keypoints import (
    read_json,
    write_to_json_file,
)

if __name__ == "__main__":

    model = load_model(
        "GroundingDINO_SwinT_OGC.py",
        "/Users/derek/Desktop/GroundingDINO/groundingdino_swint_ogc.pth",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    TEXT_PROMPT = "tennis player"
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    JSON_PATH = (
        "/Users/derek/Desktop/tennis_tracker/tennis_tracker/download_data/label.json"
    )

    data = read_json(JSON_PATH)
    img_paths = [img_path for img_path in data.keys()]

    batch_size = 10
    images = os.listdir("/Users/derek/Desktop/GroundingDINO/oct15")
    OUTPUT_JSON_PATH = JSON_PATH

    for i in tqdm(range(0, len(img_paths), batch_size)):
        batch_images = images[i : i + batch_size]
        loaded_images = []
        for image in batch_images:
            image_source, image = load_image(image)
            loaded_images.append(image)
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
            lines.append(all_boxes)
            data[batch_images[im_num]]['boxes'] = all_boxes
    write_to_json_file(OUTPUT_JSON_PATH, data)
