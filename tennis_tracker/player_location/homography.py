import cv2
import numpy as np
import torch
import torch.nn.functional as F

from tennis_tracker.player_location import BallTrackerNet, postprocess


def get_homography(
    img_path: str, model: torch.nn.Module, court_coords: list[list[int]]
):
    im = cv2.imread(img_path)

    im = cv2.resize(im, (640, 360))
    im = im.astype(np.float32) / 255.0
    im = np.rollaxis(im, 2, 0)
    im = torch.tensor(im)
    im = im.unsqueeze(0)

    out = model(im)
    pred = F.sigmoid(out).cpu().detach().numpy()[0]

    points = []
    for kps_num in range(14):
        heatmap = (pred[kps_num] * 255).astype(np.uint8)
        x_pred, y_pred = postprocess(heatmap, scale=2, low_thresh=170, max_radius=25)
        points.append((x_pred, y_pred))

    source = np.array(points)
    target = np.array(court_coords)
    m, _ = cv2.findHomography(source, target)
    return m


def get_model():
    model = BallTrackerNet(out_channels=15)
    model.load_state_dict(torch.load("model_tennis_court_det.pt", map_location="mps"))
    model.eval()
    return model


def read_court_coords(file_path: str):
    with open(file_path, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split(",") for line in lines]
    return [[int(x), int(y)] for x, y in lines]


if __name__ == "__main__":
    # could do with a dataloader once we have enough data
    pass