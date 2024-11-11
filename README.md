In the download data directory we have script to download the data (download_videos.py) extract all the frames to the frames directory (extract_frames.py) and then extract all keypoints using the tracknet model (extract_keypoints.py). The last step creates a label.jsonl file that discards any frames that do not have all keypoints thereby helping us to filter frames so we are only training on the broadcast or a good view of the court and no player shots or highlights. In that file I also find the homography matrix so in pseudo label we can predict what it will be like in the world coordinates instead of the image coordinates.

In the pseudo_label directory we have code to label the players with GroundingDino - this should allow some distillation.

Workflow should go 
download_data/download_videos.py
download_data/extract_frames.py
download_data/extract_keypoints.py
psudeo_label/pseudo_label.py


TODO
- Add ball tracking


# Workflow

1. Download the data. Navigate to the download_data directory run `python download_any_video.py`. Then to extract the frames run `python extract_frames.py`. This should extract frames for the point shown in the video to the `frames` directory.

2. Extract the court keypoints. We need to extract court keypoints in order to get the homography matrix. Navigate to the player_location directory and run `python extract_keypoints.py`. This should create a labels.json file in the labels directory.

3. Get bounding boxes for the players. Navigate to the player_tracking directory and run `python label_players.py`. This should create a labels.json file in the labels directory with all of the information from step 2 with additional bounding boxes, homography matrices, and ankle keypoint coordinates in the world coordinate system.

4. Visualize player tracking. Navigate to the visualize directory and run `python visualize_player_tracking.py`. This should create a video in the visualize directory with the player tracking both on the real image, the homography transformed image, and with both combined into one video.