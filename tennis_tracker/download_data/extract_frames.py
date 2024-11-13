import os

import cv2

# Shown Serena v Azarenka point: 31:08 - 31:30

if __name__ == "__main__":
    OUTPUT_PATH = "frames"

    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    all_videos = ["dataset/serena_v_azarenka.mp4"]

    for video_path in all_videos:
        video = cv2.VideoCapture(video_path)
        new_output_path = f"{OUTPUT_PATH}/{video_path.split('/')[-1][:-4]}"
        if not os.path.exists(new_output_path):
            os.mkdir(new_output_path)
        fps = video.get(cv2.CAP_PROP_FPS)

        print(video_path)
        # Extract frames from 31:00 to 31:30
        os.system(
            f"ffmpeg -i '{video_path}' -ss 1860 -t 30 -vf fps={fps} '{new_output_path}/%d.png'"
        )
