import cv2
import numpy as np
from tqdm import tqdm

from tennis_tracker.player_location.extract_keypoints import (
    read_court_coords, read_json_file)


def plot_keypoints(frame: np.array, keypoints: list[np.array]):
    for keypoint in keypoints:
        cv2.circle(
            frame,
            (keypoint[0], keypoint[1]),
            5,
            (0, 0, 255),
            -1,
        )
    return frame


if __name__ == "__main__":
    JSON_PATH = "../labels/labels_with_balls.json"

    data = read_json_file(JSON_PATH)

    video_frames = []
    keys = list(data.keys())
    for key in keys:
        video_frames.append((key, int(key.split("/")[-1].split(".")[0])))

    video_frames.sort(key=lambda x: x[1])
    video_frames = [x[0] for x in video_frames]

    i = 100
    frame = cv2.imread(video_frames[i])
    keypoints = data[video_frames[i]]["keypoints"]
    frame = plot_keypoints(frame, keypoints)

    court_frame = cv2.imread("../player_location/padded_court.jpg")
    court_points = read_court_coords("../player_location/padded_click_coordinates.txt")
    court_frame = plot_keypoints(court_frame, court_points)

    court_frame = cv2.resize(court_frame, (court_frame.shape[1], frame.shape[0]))

    combined_frame = np.concatenate((frame, court_frame), axis=1)
    cv2.imwrite("keypoints.png", combined_frame)
    cv2.imwrite("combined_keypoints.png", combined_frame)
