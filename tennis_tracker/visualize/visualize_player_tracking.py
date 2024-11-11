import argparse

import cv2
import numpy as np
from object_tracking.kalman_filter.schema_n_adaptive_q import (
    KalmanNDTrackerAdaptiveQ,
    KalmanStateVectorNDAdaptiveQ,
)
from tqdm import tqdm

from tennis_tracker.player_location.extract_keypoints import read_json_file

STATE = KalmanStateVectorNDAdaptiveQ
TRACKER = KalmanNDTrackerAdaptiveQ


def process_boxes(boxes: list[str]):
    """Process the boxes into a list of numpy arrays"""
    boxes = [x.split(" ") for x in boxes]
    boxes = [[float(y) for y in x[1:]] for x in boxes]
    return boxes


def extract_measurement(
    image: np.array,
    ball_tracking_boxes: list[np.array],
    transformed_coords: list[np.array],
) -> list[np.array]:
    """
    For each box returns cx, cy, w, h and the color mean.
    The center x and center y will not be normalized
    """
    measurements = []
    image_height, image_width, _ = image.shape
    for i, box in enumerate(ball_tracking_boxes):
        cx = int(box[0] * image_width)
        cy = int(box[1] * image_height)
        color_mean = get_color_mean(image, box)
        measurements.append(
            np.array(
                [
                    cx,
                    cy,
                    box[2] * image_width,
                    box[3] * image_height,
                    color_mean[0],
                    color_mean[1],
                    color_mean[2],
                    transformed_coords[i][0],
                    transformed_coords[i][1],
                ]
            )
        )
    return measurements


def process_frame(data_entry: dict) -> tuple[np.array, list[np.array]]:
    """Process a single frame of data - gets out the frame and all coordinates"""
    frame = cv2.imread(data_entry["actual_path"])
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ball_tracking_boxes = process_boxes(data_entry["boxes"])
    transformed_coords = data_entry["transformed_coords"]
    measurements = extract_measurement(
        frame_rgb, ball_tracking_boxes, transformed_coords
    )
    return frame_rgb, measurements


def get_color_mean(image: np.array, box: list[float]):
    """Get the color mean of the image"""
    image_height, image_width, _ = image.shape
    cx = int((box[0] - box[2] / 2) * image_width)
    cy = int((box[1] - box[3] / 2) * image_height)

    return np.mean(
        image[
            cy : cy + int(box[3] * image_height), cx : cx + int(box[2] * image_width)
        ],
        axis=(0, 1),
    )  # (3, )


def associate_objects(
    tracked_objects: list[tuple[KalmanNDTrackerAdaptiveQ, int]],
    detected_object_measurements: list[np.array],
    frame_number: int,
    p_value_threshold: float = 0.3,
    frame_threshold: int = 30,
    R: float = 3,
    Q: float = 3,
    h_matrix: np.array = None,
):
    """Associates detected objects with existing tracked objects using Mahalanobis distance.

    Creates new tracked objects for unmatched detections. Removes tracked objects not seen
    for frame_threshold frames.

    Args:
        tracked_objects: List of tuples containing (KalmanTracker, last_seen_frame)
        detected_object_measurements: List of measurement arrays for detected objects
        frame_number: Current frame number
        p_value_threshold: Threshold for associating detections with tracked objects
        frame_threshold: Number of frames before removing unmatched tracked objects
        R: Measurement noise parameter
        Q: Process noise parameter
        h_matrix: Optional measurement matrix

    Returns:
        Tuple containing:
        - Dictionary mapping tracked object indices to detection indices
        - List of remaining tracked objects
        - List of new tracked objects
    """
    associations = {}
    for i, (tracked_object, last_seen_frame) in enumerate(tracked_objects):
        if last_seen_frame + frame_threshold < frame_number:
            tracked_objects.pop(i)

    new_objects = []
    if len(tracked_objects) == 0:
        for detected_object_measurement in detected_object_measurements:
            vector = np.concatenate(
                [
                    detected_object_measurement,
                    np.zeros(detected_object_measurement.shape[0]),
                ]
            )
            state_vector = STATE(states=vector)
            new_objects.append(
                [
                    TRACKER(state=state_vector, R=R, Q=Q, h=h_matrix),
                    frame_number,
                ]
            )
        return {}, [], new_objects

    for i, (tracked_object, last_seen_frame) in enumerate(tracked_objects):
        tracked_object.predict(dt=1)

    for i, detected_object_measurement in enumerate(detected_object_measurements):
        max_p_value = -np.inf
        max_idx = None
        for j, (tracked_object, _) in enumerate(tracked_objects):
            p_value = tracked_object.compute_p_value_from_measurement(
                detected_object_measurement
            )
            if p_value > max_p_value:
                max_p_value = p_value
                max_idx = j
        if max_p_value > p_value_threshold:
            associations[max_idx] = i
        else:
            # now we have to initialize a new object
            # concat velocities of all the measurements to 0
            vector = np.concatenate(
                [
                    detected_object_measurement,
                    np.zeros(detected_object_measurement.shape[0]),
                ]
            )
            state_vector = STATE(states=vector)
            new_object = TRACKER(state=state_vector, R=R, Q=Q, h=h_matrix)
            new_objects.append([new_object, frame_number])

    return associations, tracked_objects, new_objects


def select_players_object(tracked_objects: list[tuple[KalmanNDTrackerAdaptiveQ, int]]):
    """Selects the players based on the most measurements"""
    objects = sorted(
        tracked_objects, key=lambda x: len(x[0].previous_measurements), reverse=True
    )
    return objects[:2]


def plot_frame(
    frame: np.array,
    tracked_object: tuple[KalmanNDTrackerAdaptiveQ, int],
    transformed: bool = False,
    kalman_coords: bool = False,
):
    """Plot the frame with the tracked objects"""

    if tracked_object is None:
        return frame
    tracked_object, _ = tracked_object
    if kalman_coords:
        measurement = tracked_object.state.state_matrix
    else:
        measurement = tracked_object.previous_measurements[-1]
    if transformed:
        # corresponds to the transformed coords
        coords = [int(measurement[7]), int(measurement[8])]
    else:
        coords = [int(measurement[0]), int(measurement[1])]
    cv2.circle(
        frame,
        coords,
        10,
        (0, 0, 255),
        -1,
    )
    return frame


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="../labels/labels.jsonl")
    parser.add_argument(
        "--video_path",
        type=str,
        default="../download_data/dataset/serena_v_azarenka.mp4",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    h_matrix = np.eye(9, 18)
    h_matrix[:, 9:] = np.eye(9)

    data = read_json_file(args.file_path)

    video_frames = []
    keys = list(data.keys())
    for key in keys:
        video_frames.append((key, int(key.split("/")[-1].split(".")[0])))

    video_frames.sort(key=lambda x: x[1])
    video_frames = [x[0] for x in video_frames]
    video_capture = cv2.VideoCapture(args.video_path)
    FPS = video_capture.get(cv2.CAP_PROP_FPS)

    current_objects = []
    # read first frame to get resolution
    _, _ = video_capture.read()
    FRAME_WIDTH = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    transformed_court_image = cv2.imread(
        "/Users/derek/Desktop/tennis_tracker/tennis_tracker/player_location/padded_court.jpg"
    )
    transformed_court_image = cv2.cvtColor(transformed_court_image, cv2.COLOR_BGR2RGB)
    COURT_HEIGHT, COURT_WIDTH, _ = transformed_court_image.shape

    output_video = cv2.VideoWriter(
        "tennis_ball_tracking_output.mp4",
        cv2.VideoWriter_fourcc(*"XVID"),
        FPS,
        (FRAME_WIDTH, FRAME_HEIGHT),
    )
    output_video_2 = cv2.VideoWriter(
        "transformed_tracking.mp4",
        cv2.VideoWriter_fourcc(*"XVID"),
        FPS,
        (COURT_WIDTH, COURT_HEIGHT),
    )
    output_video_3 = cv2.VideoWriter(
        "combined_tracking.mp4",
        cv2.VideoWriter_fourcc(*"XVID"),
        FPS,
        (FRAME_WIDTH + COURT_WIDTH, max(FRAME_HEIGHT, COURT_HEIGHT)),
    )

    for i in tqdm(range(len(video_frames))):
        frame, measurements = process_frame(data[video_frames[i]])
        court_image = transformed_court_image.copy()
        associations, current_objects, new_objects = associate_objects(
            tracked_objects=current_objects,
            detected_object_measurements=measurements,
            frame_number=i,
            h_matrix=h_matrix,
        )

        # have already predicted so just need to update
        for key, value in associations.items():
            current_objects[key][0].update(measurements[value], predict=False)
            current_objects[key][1] = i

        current_objects.extend(new_objects)

        if len(current_objects) > 1:
            # select the objects most likely to be the players
            selected_objects = select_players_object(current_objects)
            for object in selected_objects:
                frame = plot_frame(frame, object, kalman_coords=True)
                court_image = plot_frame(
                    court_image, object, transformed=True, kalman_coords=True
                )
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            court_image = cv2.cvtColor(court_image, cv2.COLOR_RGB2BGR)
            output_video.write(frame)
            output_video_2.write(court_image)
            combined_frame = np.zeros(
                (max(FRAME_HEIGHT, COURT_HEIGHT), FRAME_WIDTH + COURT_WIDTH, 3),
                dtype=np.uint8,
            )
            combined_frame[:FRAME_HEIGHT, :FRAME_WIDTH, :] = frame
            combined_frame[
                :COURT_HEIGHT, FRAME_WIDTH : FRAME_WIDTH + COURT_WIDTH, :
            ] = court_image
            output_video_3.write(combined_frame)

        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            court_image = cv2.cvtColor(court_image, cv2.COLOR_RGB2BGR)
            output_video.write(frame)
            output_video_2.write(court_image)
            combined_frame = np.zeros(
                (max(FRAME_HEIGHT, COURT_HEIGHT), FRAME_WIDTH + COURT_WIDTH, 3),
                dtype=np.uint8,
            )
            combined_frame[:FRAME_HEIGHT, :FRAME_WIDTH, :] = frame
            combined_frame[
                :COURT_HEIGHT, FRAME_WIDTH : FRAME_WIDTH + COURT_WIDTH, :
            ] = court_image
            output_video_3.write(combined_frame)

    output_video.release()
    output_video_2.release()
    output_video_3.release()
