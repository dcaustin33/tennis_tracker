import cv2
import numpy as np
from tqdm import tqdm

from tennis_tracker.player_location.extract_keypoints import read_json_file
from tennis_tracker.visualize.visualize_ball_tracking import process_frame


def plot_boxes(frame: np.array, measurements: list[np.array], plot_ankles: bool = False):
    for measurement in measurements:
        top_left_x = int(measurement[0] - measurement[2] / 2)
        top_left_y = int(measurement[1] - measurement[3] / 2)
        bottom_right_x = int(measurement[0] + measurement[2] / 2)
        bottom_right_y = int(measurement[1] + measurement[3] / 2)
        if plot_ankles:
            cv2.circle(
                frame,
                (int(measurement[0]), bottom_right_y),
                10,
                (255, 0, 0),
                -1,
            )
        else:
            cv2.rectangle(
                frame,
                (top_left_x, top_left_y),
                (bottom_right_x, bottom_right_y),
                (0, 0, 255),
                2,
            )
    return frame

def plot_keypoints(frame: np.array, measurements: list[np.array]):
    for measurement in measurements:
        cv2.circle(
            frame,
            (int(measurement[-2]), int(measurement[-1])), # Convert coordinates to integers
            5,
            (0, 0, 255),
            -1,
        )
    return frame


if __name__ == "__main__":
    JSON_PATH = "../labels/labels_with_balls.json"

    data = read_json_file(JSON_PATH)
    court_frame = cv2.imread("../player_location/padded_court.jpg")


    video_frames = []
    keys = list(data.keys())
    for key in keys:
        video_frames.append((key, int(key.split("/")[-1].split(".")[0])))

    video_frames.sort(key=lambda x: x[1])
    video_frames = [x[0] for x in video_frames]

    i = 10
    _, measurements = process_frame(data[video_frames[i]], video_frames[i])
    measurements = measurements[:2]
    frame = cv2.imread(video_frames[i])
    frame = plot_boxes(frame, measurements, plot_ankles=True)

    i = 680
    _, measurements = process_frame(data[video_frames[i]], video_frames[i])
    measurements = measurements[:2]
    frame2 = cv2.imread(video_frames[i])
    frame2 = plot_boxes(frame2, measurements, plot_ankles=True)

    frame3 = plot_keypoints(court_frame, measurements)

    frame3 = cv2.resize(frame3, (frame3.shape[1], frame2.shape[0]))
    combined_frame2 = np.concatenate((frame2, frame3), axis=1)

    # combine the two frames
    combined_frame = np.concatenate((frame, frame2), axis=1)

    cv2.imwrite("frame.png", frame)
    cv2.imwrite("frame2.png", frame2)
    cv2.imwrite("frame3.png", frame3)
    cv2.imwrite("combined_frame.png", combined_frame)
    cv2.imwrite("combined_frame2.png", combined_frame2)