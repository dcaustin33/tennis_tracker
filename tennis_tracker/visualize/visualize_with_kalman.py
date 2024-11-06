import cv2
import numpy as np
from object_tracking.kalman_filter.schema_nd import KalmanNDTracker, KalmanStateVectorND
from tqdm import tqdm

from tennis_tracker.download_data.extract_keypoints import read_json_file

FILE_PATH = (
    "/Users/derek/Desktop/tennis_tracker/tennis_tracker/pseudo_label/clean_labels.json"
)
FPS = 30
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
COURT_HEIGHT = 799
COURT_WIDTH = 560
VIDEO_PATH = "/Users/derek/Desktop/tennis_tracker/tennis_tracker/download_data/dataset/Tiafoe Takes On Evans; Khachanov& Mannarino Also In Action ｜ Almaty 2024 Highlights Day 4 [Q1iTjk444RU].webm"


def get_next_frame(
    point1: list[float],
    point2: list[float],
    measurement_point1: list[float],
    measurement_point2: list[float],
    actual_path: str,
    court_image: np.array,
):
    """We want to get tracked points and draw them on the court image"""
    original_image = cv2.imread(actual_path)
    original_image = cv2.resize(original_image, (640, 480))
    for point in [point1, point2]:
        cv2.circle(court_image, [int(p) for p in point[:2]], 10, (0, 0, 255), -1)
    # for point in [measurement_point1, measurement_point2]:
    #     if point is not None:
    #         cv2.circle(court_image, [int(p) for p in point[:2]], 10, (0, 255, 0), -1)
    return original_image


def associate_coordinates(
    tracked_points: list[list[float]],
    tracker_player1: KalmanNDTracker,
    tracker_player2: KalmanNDTracker,
):
    """
    TODO: Do this after predicting the next state. Also better matching logic

    Tracked points input here should always be the tracked points for the current frame
    """
    if len(tracked_points) == 0:
        return {0: None, 1: None}

    distance = []
    for tracker in [tracker_player1, tracker_player2]:
        tracker_distances = []
        for point in tracked_points:
            x_distance = np.linalg.norm(tracker.state.state_matrix[0] - point[0])
            y_distance = np.linalg.norm(tracker.state.state_matrix[1] - point[1])
            tracker_distances.append(x_distance + y_distance)
        distance.append(tracker_distances)

    # Get indices of minimum distances for both trackers
    min_idx_player1 = np.argmin(distance[0])
    min_idx_player2 = np.argmin(distance[1])

    # If both trackers prefer the same point, assign it to the closer tracker

    if len(tracked_points) == 1:
        if distance[0][0] <= distance[1][0]:
            return {0: tracked_points[0], 1: None}
        else:
            return {0: None, 1: tracked_points[0]}

    if min_idx_player1 == min_idx_player2:
        if distance[0][min_idx_player1] <= distance[1][min_idx_player2]:
            # Point goes to player 1, player 2 gets the other point
            return {
                0: tracked_points[min_idx_player1],
                1: tracked_points[1 - min_idx_player1],
            }
        else:
            # Point goes to player 2, player 1 gets the other point
            return {
                0: tracked_points[1 - min_idx_player1],
                1: tracked_points[min_idx_player1],
            }
    else:
        # Each tracker gets their preferred point
        return {0: tracked_points[min_idx_player1], 1: tracked_points[min_idx_player2]}


if __name__ == "__main__":
    data = read_json_file(FILE_PATH)

    tiafoe_video_frames = []
    keys = list(data.keys())
    for key in keys:
        if "Tiafoe" in key:
            tiafoe_video_frames.append((key, int(key.split("/")[-1].split(".")[0])))

    tiafoe_video_frames.sort(key=lambda x: x[1])
    tiafoe_video_frames = [x[0] for x in tiafoe_video_frames]
    video_capture = cv2.VideoCapture(VIDEO_PATH)
    FPS = video_capture.get(cv2.CAP_PROP_FPS)

    output_video = cv2.VideoWriter(
        "tennis_tracking_kalman_output.mp4",
        cv2.VideoWriter_fourcc(*"XVID"),
        FPS,
        (FRAME_WIDTH + COURT_WIDTH, max(FRAME_HEIGHT, COURT_HEIGHT)),
    )
    tracker_player1 = None

    tracker_player2 = None
    for i in tqdm(range(len(tiafoe_video_frames[:3000]))):
        court_image = cv2.imread(
            "/Users/derek/Desktop/tennis_tracker/tennis_tracker/player_location/padded_court.jpg"
        )
        tracked_points = [
            [int(p[0]), int(p[1])]
            for p in data[tiafoe_video_frames[i]]["transformed_coords"]
        ]
        if tracker_player1 is None:
            tracker_player1 = KalmanNDTracker(
                state=KalmanStateVectorND(
                    states=np.array([tracked_points[0][0], tracked_points[0][1]]),
                    velocities=np.array([0, 0]),
                ),
                state_noise_std=1,
                measurement_noise_std=1,
                h=np.array([[1, 0, 0, 0], [0, 1, 0, 0]]),
            )
            tracker_player2 = KalmanNDTracker(
                state=KalmanStateVectorND(
                    states=np.array([tracked_points[1][0], tracked_points[1][1]]),
                    velocities=np.array([0, 0]),
                ),
                state_noise_std=1,
                measurement_noise_std=1,
                h=np.array([[1, 0, 0, 0], [0, 1, 0, 0]]),
            )
        else:
            assignments = associate_coordinates(
                tracked_points, tracker_player1, tracker_player2
            )
            if assignments[0] is not None:
                tracker_player1.update(np.array([assignments[0][0], assignments[0][1]]))
            if assignments[1] is not None:
                tracker_player2.update(np.array([assignments[1][0], assignments[1][1]]))

        if len(tracked_points) == 0:
            measurement_point1 = None
            measurement_point2 = None
        elif len(tracked_points) == 1:
            measurement_point1 = tracked_points[0]
            measurement_point2 = None
        else:
            measurement_point1 = tracked_points[0]
            measurement_point2 = tracked_points[1]

        original_frame = get_next_frame(
            point1=tracker_player1.state.state_matrix[:2],
            point2=tracker_player2.state.state_matrix[:2],
            measurement_point1=measurement_point1,
            measurement_point2=measurement_point2,
            actual_path=data[tiafoe_video_frames[i]]["actual_path"],
            court_image=court_image,
        )

        frame = np.zeros(
            (max(FRAME_HEIGHT, COURT_HEIGHT), FRAME_WIDTH + COURT_WIDTH, 3),
            dtype=np.uint8,
        )
        frame[:FRAME_HEIGHT, :FRAME_WIDTH, :] = original_frame
        frame[:COURT_HEIGHT, FRAME_WIDTH : FRAME_WIDTH + COURT_WIDTH, :] = court_image
        # cv2.imshow("Tennis Tracking", frame)
        output_video.write(frame)
        # if cv2.waitKey(int(1000 / FPS)) & 0xFF == ord("q"):
        #     break
