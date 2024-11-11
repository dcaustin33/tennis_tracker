import argparse
import json
import os
import pathlib
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data.dataloader
from PIL import Image
from tqdm import tqdm

from tennis_tracker.player_location.postprocess import postprocess
from tennis_tracker.player_location.tracknet import BallTrackerNet


class tracknet_transform(torch.nn.Module):
    def __init__(
        self,
        size: tuple,
    ) -> None:
        self.img_size = size

    def __call__(self, img_path: str) -> torch.Tensor:
        image = Image.open(img_path).convert("RGB").resize(self.img_size)
        image = torch.from_numpy(np.array(image) / 255.0).float()
        image = image.permute(2, 0, 1)
        return image


class frame_dataset(torch.utils.data.Dataset):

    def __init__(self, frames_directory: str, size: tuple = (640, 360)) -> None:
        self.all_image_paths = []
        for dir_path, sub_dirs, files in os.walk(frames_directory):
            for file in sorted(files):
                if file.endswith(".png"):
                    self.all_image_paths.append(os.path.join(dir_path, file))
        self.transform = tracknet_transform(size)

    def __len__(
        self,
    ) -> int:
        return len(self.all_image_paths)

    def __getitem__(self, idx: int) -> tuple:
        """Will return a tuple of the image path and torch tensor image"""
        img_path = self.all_image_paths[idx]
        return self.transform(img_path), img_path


def write_to_jsonl_file(file_path: str, lines: list) -> None:
    if not os.path.exists(file_path):
        mode = "w"
    else:
        mode = "a"

    with open(file_path, mode=mode) as f:
        for line in lines:
            json.dump(line, f, ensure_ascii=False)
            f.write("\n")


def read_jsonl(
    file_path: str,
) -> list:
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def write_to_json_file(file_path: str, data: dict) -> None:
    if not pathlib.Path(file_path).exists():
        with open(file_path, "w") as f:
            json.dump({}, f)

    with open(file_path, "r+") as f:
        existing_data = json.load(f)

        existing_data.update(data)
        f.seek(0)
        json.dump(existing_data, f, indent=4, ensure_ascii=False)
        f.truncate()


def read_json_file(file_path: str) -> dict:
    with open(file_path, "r+") as f:
        existing_data = json.load(f)
    return existing_data


def read_court_coords(file_path: str):
    with open(file_path, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split(",") for line in lines]
    return [[int(x), int(y)] for x, y in lines]


def get_visible_points(points: list, source_points: list) -> list:
    visible_points = []
    visible_source_points = []
    for point, source_point in zip(points, source_points):

        if point != (None, None):
            visible_points.append(point)
            visible_source_points.append(source_point)
    return visible_points, visible_source_points


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./model_tennis_court_det.pt")
    parser.add_argument(
        "--dataset_path", type=str, default="../download_data/frames/serena_v_azarenka"
    )
    parser.add_argument(
        "--output_json_path", type=str, default="../labels/labels.json"
    )
    parser.add_argument(
        "--court_coordinates_path", type=str, default="./padded_click_coordinates.txt"
    )
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # ARGS
    args = parse_args()
    MODEL_PATH = args.model_path
    DATASET_PATH = args.dataset_path
    OUTPUT_JSON_PATH = args.output_json_path
    COURT_COORDINATES_PATH = args.court_coordinates_path
    BATCH_SIZE = args.batch_size
    DEVICE = args.device

    model = BallTrackerNet(out_channels=15)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    model = model.to(DEVICE)

    dataset = frame_dataset(DATASET_PATH)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6
    )
    print(f"The dataloader will have {len(dataset) / BATCH_SIZE} steps")
    lines = []
    lines = {}

    if os.path.exists(OUTPUT_JSON_PATH):
        os.remove(OUTPUT_JSON_PATH)

    time_now = time.time()
    # we are also going to get the homography matrix for each frame
    court_coordinates = np.array(read_court_coords(COURT_COORDINATES_PATH))

    for idx, batch in tqdm(enumerate(dataloader)):
        imgs, img_paths = batch
        imgs = imgs.to(DEVICE)
        with torch.no_grad():
            output = model(imgs)
        preds = F.sigmoid(output).cpu().detach().numpy()
        all_preds = []
        paths = []
        for pred_idx in range(preds.shape[0]):
            pred = preds[pred_idx]
            points = []
            for kps_num in range(14):
                heatmap = (pred[kps_num] * 255).astype(np.uint8)
                x_pred, y_pred = postprocess(
                    heatmap, scale=2, low_thresh=170, max_radius=25
                )
                points.append((x_pred, y_pred))

            # if the tracknet produces None means it is not visible / not there
            # we are using this as our filtering criteria so we only capture frames with
            # points present
            if (None, None) in points:
                # check to see if at least 10 are visible
                if len([p for p in points if p != (None, None)]) <= 10:
                    continue
                points, visible_court_coords = get_visible_points(
                    points, court_coordinates
                )

                m, _ = cv2.findHomography(
                    np.array(points), np.array(visible_court_coords)
                )
            else:
                m, _ = cv2.findHomography(np.array(points), court_coordinates)
            lines[img_paths[pred_idx]] = {
                "keypoints": [[int(point[0]), int(point[1])] for point in points],
                "image_dims": imgs[pred_idx].shape[-2:][::-1],
                "m": m.tolist(),
            }
        write_to_json_file(OUTPUT_JSON_PATH, lines)
        lines = {}
