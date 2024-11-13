import cv2
import numpy as np
from tqdm import tqdm

from tennis_tracker.player_location.extract_keypoints import read_json_file
from tennis_tracker.visualize.visualize_ball_tracking import process_frame


def plot_boxes(frame: np.array, measurements: list[np.array]):
    for measurement in measurements:
        top_left_x = int(measurement[0] - measurement[2] / 2)
        top_left_y = int(measurement[1] - measurement[3] / 2)
        bottom_right_x = int(measurement[0] + measurement[2] / 2)
        bottom_right_y = int(measurement[1] + measurement[3] / 2)
        cv2.rectangle(
            frame,
            (top_left_x, top_left_y),
            (bottom_right_x, bottom_right_y),
            (0, 0, 255),
            2,
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

    i = 10
    _, measurements = process_frame(
        data[video_frames[i]], video_frames[i]
    )
    frame = cv2.imread(video_frames[i])
    frame = plot_boxes(frame, measurements)
    
    i = 80
    _, measurements = process_frame(
        data[video_frames[i]], video_frames[i]
    )
    frame2 = cv2.imread(video_frames[i])
    frame2 = plot_boxes(frame2, measurements)
    
    # combine the two frames
    combined_frame = np.concatenate((frame, frame2), axis=1)
    
    cv2.imwrite("frame.png", frame)
    cv2.imwrite("frame2.png", frame2)
    cv2.imwrite("combined_frame.png", combined_frame)
