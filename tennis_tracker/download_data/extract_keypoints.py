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


if __name__ == "__main__":
    # ARGS
    model_path = "/Users/derek/Desktop/tennis_tracker/tennis_tracker/player_location/model_tennis_court_det.pt"
    dataset_path = (
        "/Users/derek/Desktop/tennis_tracker/tennis_tracker/download_data/frames"
    )
    json_file_path = (
        "/Users/derek/Desktop/tennis_tracker/tennis_tracker/download_data/labels.json"
    )
    court_coordinates_path = "/Users/derek/Desktop/tennis_tracker/tennis_tracker/player_location/padded_click_coordinates.txt"
    batch_size = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BallTrackerNet(out_channels=15)
    model.load_state_dict(torch.load(model_path, map_location="mps"))
    model.eval()
    model = torch.compile(model)
    model = model.to(device)

    dataset = frame_dataset(dataset_path)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=6
    )
    print(f"The dataloader will have {len(dataset) / batch_size} steps")
    lines = []
    lines = {}

    if os.path.exists(json_file_path):
        os.remove(json_file_path)

    time_now = time.time()
    # we are also going to get the homography matrix
    court_coordinates = np.array(read_court_coords(court_coordinates_path))

    for idx, batch in enumerate(dataloader):
        if idx % 10 == 0:
            print(idx, time.time() - time_now)
            time_now = time.time()
        imgs, img_paths = batch
        imgs = imgs.to(device)
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
            # we are using this as our filtering criteria so we only capture good points
            if (None, None) in points:
                continue
            m, _ = cv2.findHomography(np.array(points), court_coordinates)
            lines[img_paths[pred_idx]] = {
                "keypoints": [[int(point[0]), int(point[1])] for point in points],
                "image_dims": imgs[pred_idx].shape[-2:],
                "m": m.tolist(),
            }
        # if len(lines) > 100:
        write_to_json_file(json_file_path, lines)
        lines = {}
