

import cv2
import numpy as np
from object_tracking.kalman_filter.schema_n_adaptive_q import (
    KalmanNDTrackerAdaptiveQ,
    KalmanStateVectorNDAdaptiveQ,
)
from tqdm import tqdm

from tennis_tracker.download_data.extract_keypoints import read_json_file

STATE = KalmanStateVectorNDAdaptiveQ
TRACKER = KalmanNDTrackerAdaptiveQ


def process_boxes(boxes: list[str]):
    """Process the boxes into a list of numpy arrays"""
    boxes = [x.split(" ") for x in boxes]
    boxes = [[float(y) for y in x[1:]] for x in boxes]
    return boxes


def extract_measurement(
    image: np.array, ball_tracking_boxes: list[np.array]
) -> list[np.array]:
    """
    For each box returns cx, cy, w, h and the color mean.
    The center x and center y will not be normalized
    """
    measurements = []
    image_height, image_width, _ = image.shape
    for box in ball_tracking_boxes:
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
                ]
            )
        )
    return measurements


def process_frame(data_entry: dict) -> tuple[np.array, list[np.array]]:
    """Process a single frame of data - gets out the frame and all coordinates"""
    frame = cv2.imread(data_entry["actual_path"])
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ball_tracking_boxes = process_boxes(data_entry["boxes"])
    measurements = extract_measurement(frame_rgb, ball_tracking_boxes)
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
    p_value_threshold: float = 0.05,
    frame_threshold: int = 100,
    R: float = 3,
    Q: float = 3,
    h_matrix: np.array = None,
):
    """Associate the predicted objects with the detected objects

    Ok so lets think through this association
    For each object we have we want to predict the next state
    Then we want to compute the mahalanobis distance between the predicted state and each of the detected states
    If it is less than some threshold we will associate the two
    If there are no associations we will create a new object.

    Adding the measurement to the previous measurements will be done in the update function.
    Tracked objects should be a tuple with the last time we saw the object and drop if more than 100 frames have passed.
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
                (
                    TRACKER(state=state_vector, R=R, Q=Q, h=h_matrix),
                    frame_number,
                )
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
            new_objects.append((new_object, frame_number))

    return associations, current_objects, new_objects


def select_players_object(tracked_objects: list[tuple[KalmanNDTrackerAdaptiveQ, int]]):
    """Select the ball based on most yellow color and less than 10 width and height"""
    objects = sorted(tracked_objects, key=lambda x: len(x[0].previous_measurements))
    return objects[0]


def plot_frame(frame: np.array, tracked_object: tuple[KalmanNDTrackerAdaptiveQ, int]):
    """Plot the frame with the tracked objects"""
    
    if tracked_object is None:
        return frame
    tracked_object, _ = tracked_object
    measurement = tracked_object.previous_measurements[-1]
    cv2.circle(
        frame,
        [
            int(measurement[0]),
            int(measurement[1]),
        ],
        10,
        (0, 0, 255),
        -1,
    )
    # plot line from previous measurement to current measurement
    if len(tracked_object.previous_measurements) < 2:
        return frame
    previous_measurement = tracked_object.previous_measurements[-2]
    cv2.line(
        frame,
        [int(previous_measurement[0]), int(previous_measurement[1])],
        [
            int(tracked_object.state.state_matrix[0]),
            int(tracked_object.state.state_matrix[1]),
        ],
        (0, 0, 255),
        2,
    )
    return frame


def plot_detected_objects(
    frame: np.array, detected_object_measurements: list[np.array]
):
    """Plot the detected objects on the frame"""
    for detected_object_measurement in detected_object_measurements:
        cv2.circle(
            frame,
            [int(detected_object_measurement[0]), int(detected_object_measurement[1])],
            10,
            (0, 255, 0),
            -1,
        )
    return frame


def print_obj(tracked_objects: list[tuple[KalmanNDTrackerAdaptiveQ, int]]):
    """Print the current object locations - Debugging func"""
    for tracked_object, _ in tracked_objects:
        print(
            tracked_object.state.state_matrix[0],
            tracked_object.state.state_matrix[1],
            _,
        )


def print_measurements(measurements: list[np.array]):
    """Print the current measurements"""
    for measurement in measurements:
        print(measurement[0], measurement[1])


def plot_frame_ticks(frame: np.array):
    """Plot the frame with ticks showing resolution for debugging"""
    # Add ticks showing resolution
    height, width = frame.shape[:2]
    tick_spacing = 100  # Show ticks every 100 pixels

    # Add horizontal ticks and labels
    for x in range(0, width, tick_spacing):
        cv2.line(frame, (x, height - 20), (x, height - 10), (255, 255, 255), 1)
        cv2.putText(
            frame,
            str(x),
            (x - 10, height - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

    # Add vertical ticks and labels

    for y in range(0, height, tick_spacing):
        cv2.line(frame, (10, y), (20, y), (255, 255, 255), 1)
        cv2.putText(
            frame,
            str(y),
            (25, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
    return frame


if __name__ == "__main__":
    FILE_PATH = (
        "/Users/derek/Desktop/tennis_tracker/tennis_tracker/ball_tracking/clean_labels_V010_v2.json"
    )
    VIDEO_PATH = "/Users/derek/Desktop/tennis_tracker/tennis_tracker/download_data/dataset/V010.mp4"

    h_matrix = np.eye(7, 14)
    h_matrix[:, 7:] = np.eye(7)

    data = read_json_file(FILE_PATH)

    video_frames = []
    keys = list(data.keys())
    for key in keys:
        video_frames.append((key, int(key.split("/")[-1].split(".")[0])))

    video_frames.sort(key=lambda x: x[1])
    video_frames = [x[0] for x in video_frames]
    video_capture = cv2.VideoCapture(VIDEO_PATH)
    FPS = video_capture.get(cv2.CAP_PROP_FPS)

    current_objects = []
    # read first frame to get resolution
    _, _ = video_capture.read()
    FRAME_WIDTH = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_video = cv2.VideoWriter(
        "tennis_ball_tracking_output.mp4",
        cv2.VideoWriter_fourcc(*"XVID"),
        FPS,
        (FRAME_WIDTH, FRAME_HEIGHT),
    )

    offset = 0
    for i in tqdm(range(len(video_frames[offset:3000]))):
        i += offset
        frame, measurements = process_frame(data[video_frames[i]])

        associations, current_objects, new_objects = associate_objects(
            tracked_objects=current_objects,
            detected_object_measurements=measurements,
            frame_number=i,
            h_matrix=h_matrix,
        )

        # have already predicted so just need to update
        for key, value in associations.items():
            current_objects[key][0].update(measurements[value], predict=False)


        current_objects.extend(new_objects)
        import pdb; pdb.set_trace()

        if len(current_objects) > 1:
            frame = plot_detected_objects(frame, measurements)
            selected_object = select_players_object(current_objects)
            frame = plot_frame(frame, selected_object)
            frame = plot_frame_ticks(frame)
            output_video.write(frame)
            cv2.imshow("Tennis Tracking", frame)
            cv2.waitKey(int(1000 / FPS))
        else:
            output_video.write(frame)

    output_video.release()