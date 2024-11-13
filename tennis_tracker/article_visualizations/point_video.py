import cv2
from tqdm import tqdm

from tennis_tracker.player_location.extract_keypoints import read_json_file

if __name__ == "__main__":
    JSON_PATH = "../labels/labels_with_balls.json"
    VIDEO_PATH = "../download_data/dataset/serena_v_azarenka.mp4"

    data = read_json_file(JSON_PATH)

    video_frames = []
    keys = list(data.keys())
    for key in keys:
        video_frames.append((key, int(key.split("/")[-1].split(".")[0])))

    video_frames.sort(key=lambda x: x[1])
    video_frames = [x[0] for x in video_frames]
    video_capture = cv2.VideoCapture(VIDEO_PATH)
    FPS = video_capture.get(cv2.CAP_PROP_FPS)
    _, _ = video_capture.read()
    FRAME_WIDTH = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = cv2.VideoWriter(
        "point.mp4",
        cv2.VideoWriter_fourcc(*"XVID"),
        FPS,
        (FRAME_WIDTH, FRAME_HEIGHT),
    )

    for i in tqdm(range(len(video_frames))):
        frame = cv2.imread(video_frames[i])
        output_video.write(frame)
    output_video.release()
