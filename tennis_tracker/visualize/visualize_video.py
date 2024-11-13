import cv2
import numpy as np
from tqdm import tqdm

from tennis_tracker.player_location.extract_keypoints import read_json_file

FILE_PATH = (
    "/Users/derek/Desktop/tennis_tracker/tennis_tracker/pseudo_label/labels_V010.json"
)
FPS = 30
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
COURT_HEIGHT = 799
COURT_WIDTH = 560
# VIDEO_PATH = "/Users/derek/Desktop/tennis_tracker/tennis_tracker/download_data/dataset/Tiafoe Takes On Evans; Khachanov& Mannarino Also In Action ｜ Almaty 2024 Highlights Day 4 [Q1iTjk444RU].webm"
VIDEO_PATH = (
    "/Users/derek/Desktop/tennis_tracker/tennis_tracker/download_data/dataset/V010.mp4"
)


def get_next_frame(data_entry: dict, court_image: np.array):
    """We want to get tracked points and draw them on the court image"""
    tracked_points = data_entry["transformed_coords"]
    original_image = cv2.imread(data_entry["actual_path"])
    original_image = cv2.resize(original_image, (640, 480))
    for point in tracked_points:
        cv2.circle(court_image, [int(p) for p in point], 10, (0, 0, 255), -1)
    return original_image, tracked_points


if __name__ == "__main__":
    data = read_json_file(FILE_PATH)

    video_frames = []
    keys = list(data.keys())
    for key in keys:
        video_frames.append((key, int(key.split("/")[-1].split(".")[0])))

    video_frames.sort(key=lambda x: x[1])
    video_frames = [x[0] for x in video_frames]
    video_capture = cv2.VideoCapture(VIDEO_PATH)
    FPS = video_capture.get(cv2.CAP_PROP_FPS)

    output_video = cv2.VideoWriter(
        "tennis_tracking_output.mp4",
        cv2.VideoWriter_fourcc(*"XVID"),
        FPS,
        (FRAME_WIDTH + COURT_WIDTH, max(FRAME_HEIGHT, COURT_HEIGHT)),
    )
    for i in tqdm(range(len(video_frames))):
        court_image = cv2.imread(
            "/Users/derek/Desktop/tennis_tracker/tennis_tracker/player_location/padded_court.jpg"
        )
        original_frame, tracked_points = get_next_frame(
            data[video_frames[i]], court_image
        )
        frame = np.zeros(
            (max(FRAME_HEIGHT, COURT_HEIGHT), FRAME_WIDTH + COURT_WIDTH, 3),
            dtype=np.uint8,
        )
        frame[:FRAME_HEIGHT, :FRAME_WIDTH, :] = original_frame
        frame[:COURT_HEIGHT, FRAME_WIDTH : FRAME_WIDTH + COURT_WIDTH, :] = court_image
        output_video.write(frame)
        cv2.imshow("Tennis Tracking", frame)

        if cv2.waitKey(int(1000 / FPS)) & 0xFF == ord("q"):
            break
