import os


def read_paths(file_path: str):
    with open(file_path, "r") as f:
        lines = f.readlines()
    return lines

def download_video(url: str, output_path: str):
    
    command = f"wget -O {output_path} {url}"
    os.system(command)


if __name__ == "__main__":
    DATASET_PATH = "dataset"
    
    if not os.path.exists(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    lines = read_paths("serena_vid.txt")

    for line in lines:
        download_video(line, f"{DATASET_PATH}/serena_v_azarenka.mp4")
    