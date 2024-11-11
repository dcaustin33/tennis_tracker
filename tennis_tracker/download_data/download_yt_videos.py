import os

from yt_dlp import YoutubeDL


def read_paths(file_path: str):
    with open(file_path, "r") as f:
        lines = f.readlines()
    return lines


lines = read_paths("youtube_vids.txt")

ydl_opts = {}
for line in lines:
    files_before = os.listdir("./")
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([line])
    files_after = os.listdir("./")
    new_file = set(files_after).difference(files_before)
    # move new file to new folder
    file = new_file.pop()
    os.rename(file, f"dataset/{file}")
