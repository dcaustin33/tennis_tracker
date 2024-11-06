import os

import cv2

# good serena points 31:08 - 31:30

if __name__ == "__main__":
    # path_to_videos = (
    #     "/home/da2986/tennis_tracker/tennis_tracker/download_data/dataset"
    # )
    # all_videos = [os.path.join(path_to_videos, x) for x in os.listdir(path_to_videos)]

    all_videos = [
        "/Users/derek/Desktop/tennis_tracker/tennis_tracker/download_data/dataset/V010.mp4"
    ]

    for video_path in all_videos[:1]:
        video = cv2.VideoCapture(video_path)
        new_output_path = f"frames2/{video_path.split('/')[-1][:-5]}"
        if not os.path.exists(new_output_path):
            os.mkdir(new_output_path)
        fps = video.get(cv2.CAP_PROP_FPS)

        print(video_path)
        # Extract frames from 10min to 30min (600s to 1800s)
        os.system(f"ffmpeg -i '{video_path}' -ss 1860 -t 30 -vf fps={fps} '{new_output_path}/%d.png'")
