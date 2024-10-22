In the download data directory we have script to download the data (download_videos.py) extract all the frames to the frames directory (extract_frames.py) and then extract all keypoints using the tracknet model (extract_keypoints.py). The last step creates a label.jsonl file that discards any frames that do not have all keypoints thereby helping us to filter frames so we are only training on the broadcast or a good view of the court and no player shots or highlights. In that file I also find the homography matrix so in pseudo label we can predict what it will be like in the world coordinates instead of the image coordinates.

In the pseudo_label directory we have code to label the players with GroundingDino - this should allow some distillation.

Workflow should go 
download_data/download_videos.py
download_data/extract_frames.py
download_data/extract_keypoints.py
psudeo_label/pseudo_label.py


TODO
- Add ball tracking