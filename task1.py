import cv2
import numpy as np
from ultralytics import YOLO
import os


VIDEO_PATH = "C:/Users/sandh/OneDrive/Desktop/computer_vision_assignment/task1_pick_place/input/backpack_video.mp4"

OUTPUT_PATH = "C:/Users/sandh/OneDrive/Desktop/computer_vision_assignment/task1_pick_place/output/result.mp4"
           
POSE_CONF = 0.4
BACKPACK_CONF = 0.25

ENTER_FRAMES = 3
PICK_FRAMES = 3
RELEASE_FRAMES = 5
LEAVE_FRAMES = 15
RETURN_FRAMES = 3

WRIST_BAG_DISTANCE = 180
RELEASE_DISTANCE = 170

BAG_MOVEMENT_THRESHOLD = 50

WRIST_MEMORY_FRAMES = 5
BAG_MEMORY_FRAMES = 8
PERSON_MEMORY_FRAMES = 5

ROI_OVERLAP_THRESHOLD = 0.10


os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

pose_model = YOLO("yolo11n-pose.pt")
object_model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open video: " + VIDEO_PATH
    )


ret, first_frame = cap.read()


if not ret:

    raise RuntimeError(
        "Could not read first frame"
    )

print()
print("========================================")
print("SELECT BACKPACK ROI")
print("========================================")
print()
print("Draw a rectangle around the backpack.")
print("Make the ROI slightly larger than the")
print("initial backpack location.")
print()
print("Press ENTER after selecting the ROI.")
print()


roi = cv2.selectROI(
    "Select Backpack ROI",
    first_frame,
    showCrosshair=True,
    fromCenter=False
)


cv2.destroyWindow(
    "Select Backpack ROI"
)


ROI_X1, ROI_Y1, ROI_W, ROI_H = roi


ROI_X2 = ROI_X1 + ROI_W
ROI_Y2 = ROI_Y1 + ROI_H


if ROI_W == 0 or ROI_H == 0:

    raise RuntimeError(
        "ROI was not selected"
    )


print(
    f"ROI: ({ROI_X1}, {ROI_Y1}) "
    f"to ({ROI_X2}, {ROI_Y2})"
)


cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    0
)


fps = cap.get(
    cv2.CAP_PROP_FPS
)


if fps <= 0:

    fps = 30


width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)


height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)


fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)


if not writer.isOpened():

    raise RuntimeError(
        "Could not create output video"
    )


state = "PLACED"
task_completed = False


wrist_inside_count = 0
bag_moving_count = 0
release_count = 0
leave_count = 0
return_count = 0

wrist_lost_count = 0
bag_lost_count = 0
person_lost_count = 0

initial_bag_center = None
last_bag_center = None
last_bag_box = None
last_wrist_point = None
last_person_center = None


def get_center(box):

    x1, y1, x2, y2 = box

    return (
        int((x1 + x2) / 2),
        int((y1 + y2) / 2)
    )


def inside_roi(point):

    if point is None:

        return False


    x, y = point


    return (
        ROI_X1 <= x <= ROI_X2
        and
        ROI_Y1 <= y <= ROI_Y2
    )


def box_overlap_with_roi(box):

    if box is None:

        return 0.0


    x1, y1, x2, y2 = box


    ix1 = max(
        x1,
        ROI_X1
    )

    iy1 = max(
        y1,
        ROI_Y1
    )

    ix2 = min(
        x2,
        ROI_X2
    )

    iy2 = min(
        y2,
        ROI_Y2
    )


    intersection_width = max(
        0,
        ix2 - ix1
    )

    intersection_height = max(
        0,
        iy2 - iy1
    )


    intersection_area = (
        intersection_width
        *
        intersection_height
    )


    box_area = max(
        1,
        (x2 - x1) * (y2 - y1)
    )


    return (
        intersection_area
        /
        box_area
    )


def backpack_in_roi(box):

    if box is None:

        return False


    overlap = box_overlap_with_roi(
        box
    )


    return (
        overlap >= ROI_OVERLAP_THRESHOLD
    )


def distance(point1, point2):

    if (
        point1 is None
        or
        point2 is None
    ):

        return float("inf")


    return np.sqrt(
        (point1[0] - point2[0]) ** 2
        +
        (point1[1] - point2[1]) ** 2
    )


def get_wrists(keypoints):

    if keypoints is None:

        return []


    wrists = []


    if len(keypoints) > 9:

        x = keypoints[9][0]
        y = keypoints[9][1]


        if x > 0 and y > 0:

            wrists.append(
                (int(x), int(y))
            )


    if len(keypoints) > 10:

        x = keypoints[10][0]
        y = keypoints[10][1]


        if x > 0 and y > 0:

            wrists.append(
                (int(x), int(y))
            )


    return wrists


def closest_wrist(
    wrists,
    bag_center
):

    if not wrists:

        return None


    if bag_center is None:

        return wrists[0]


    return min(
        wrists,
        key=lambda p: distance(
            p,
            bag_center
        )
    )


while True:

    ret, frame = cap.read()


    if not ret:

        break

    previous_bag_center = last_bag_center

    person_found = False
    bag_found = False

    person_center = None
    bag_center = None
    wrist_point = None

    person_id = None
    bag_id = None

    wrists = []


    # =========================================
    # PERSON DETECTION + POSE + TRACKING
    # =========================================

    pose_results = pose_model.track(
        frame,
        persist=True,
        classes=[0],
        conf=POSE_CONF,
        verbose=False
    )


    if pose_results:

        result = pose_results[0]


        if (
            result.boxes is not None
            and
            len(result.boxes) > 0
        ):

            best_index = None
            largest_area = 0


            for i, box in enumerate(
                result.boxes
            ):

                xyxy = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )


                x1, y1, x2, y2 = xyxy


                area = (
                    (x2 - x1)
                    *
                    (y2 - y1)
                )


                if area > largest_area:

                    largest_area = area
                    best_index = i


            if best_index is not None:

                person_box = (
                    result
                    .boxes[best_index]
                    .xyxy[0]
                    .cpu()
                    .numpy()
                )


                person_center = get_center(
                    person_box
                )


                person_found = True


                person_lost_count = 0


                last_person_center = (
                    person_center
                )


                if (
                    result
                    .boxes[best_index]
                    .id
                    is not None
                ):

                    person_id = int(
                        result
                        .boxes[best_index]
                        .id[0]
                        .cpu()
                        .item()
                    )


                if result.keypoints is not None:

                    keypoints = (
                        result
                        .keypoints[best_index]
                        .xy[0]
                        .cpu()
                        .numpy()
                    )


                    wrists = get_wrists(
                        keypoints
                    )


                    detected_wrist = (
                        closest_wrist(
                            wrists,
                            bag_center
                        )
                    )


                    if detected_wrist is not None:

                        wrist_point = (
                            detected_wrist
                        )

                        last_wrist_point = (
                            detected_wrist
                        )

                        wrist_lost_count = 0

                    else:

                        wrist_lost_count += 1


    # =========================================
    # PERSON TEMPORARY DETECTION LOSS
    # =========================================

    if not person_found:

        person_lost_count += 1


    # =========================================
    # BACKPACK DETECTION + TRACKING
    # =========================================

    backpack_results = object_model.track(
        frame,
        persist=True,
        classes=[24],
        conf=BACKPACK_CONF,
        verbose=False
    )


    if backpack_results:

        result = backpack_results[0]


        if (
            result.boxes is not None
            and
            len(result.boxes) > 0
        ):

            selected_box = None
            selected_score = float("inf")


            for box in result.boxes:

                xyxy = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )


                center = get_center(
                    xyxy
                )


                if last_bag_center is not None:

                    score = distance(
                        center,
                        last_bag_center
                    )

                elif initial_bag_center is not None:

                    score = distance(
                        center,
                        initial_bag_center
                    )

                else:

                    score = 0


                if score < selected_score:

                    selected_score = score
                    selected_box = box


            if selected_box is not None:

                detected_bag_box = (
                    selected_box
                    .xyxy[0]
                    .cpu()
                    .numpy()
                )


                detected_bag_center = (
                    get_center(
                        detected_bag_box
                    )
                )


                bag_center = (
                    detected_bag_center
                )


                bag_found = True


                last_bag_center = (
                    detected_bag_center
                )


                last_bag_box = (
                    detected_bag_box.copy()
                )


                bag_lost_count = 0


                if initial_bag_center is None:

                    initial_bag_center = (
                        detected_bag_center
                    )


                if selected_box.id is not None:

                    bag_id = int(
                        selected_box
                        .id[0]
                        .cpu()
                        .item()
                    )


                x1, y1, x2, y2 = map(
                    int,
                    detected_bag_box
                )


                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    3
                )


                cv2.putText(
                    frame,
                    "BACKPACK",
                    (
                        x1,
                        max(
                            20,
                            y1 - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )


                if bag_id is not None:

                    cv2.putText(
                        frame,
                        f"Bag ID: {bag_id}",
                        (
                            x1,
                            y2 + 20
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2
                    )


    # =========================================
    # BACKPACK TEMPORARY DETECTION LOSS
    # =========================================

    if not bag_found:

        bag_lost_count += 1


    # =========================================
    # USE LAST KNOWN BACKPACK POSITION
    # =========================================

    effective_bag_center = bag_center
    effective_bag_box = last_bag_box


    if (
        bag_center is None
        and
        last_bag_center is not None
        and
        bag_lost_count <= BAG_MEMORY_FRAMES
    ):

        effective_bag_center = (
            last_bag_center
        )


    # =========================================
    # WRIST SELECTION
    # =========================================

    if bag_center is not None:

        current_wrist = closest_wrist(
            wrists,
            bag_center
        )


        if current_wrist is not None:

            wrist_point = current_wrist

            last_wrist_point = (
                current_wrist
            )

            wrist_lost_count = 0

        else:

            wrist_lost_count += 1


    elif wrists:

        wrist_point = wrists[0]

        last_wrist_point = (
            wrist_point
        )

        wrist_lost_count = 0


    # =========================================
    # TEMPORARY WRIST LOSS MEMORY
    # =========================================

    if (
        wrist_point is None
        and
        last_wrist_point is not None
        and
        wrist_lost_count <= WRIST_MEMORY_FRAMES
    ):

        wrist_point = (
            last_wrist_point
        )


    # =========================================
    # ROI CHECKS
    # =========================================

    wrist_inside = inside_roi(
        wrist_point
    )


    bag_inside = backpack_in_roi(
        effective_bag_box
    )


    roi_overlap = box_overlap_with_roi(
        effective_bag_box
    )


    # =========================================
    # WRIST-BACKPACK INTERACTION
    # =========================================

    wrist_near_bag = False

    wrist_distance = float("inf")


    if (
        wrist_point is not None
        and
        effective_bag_center is not None
    ):

        wrist_distance = distance(
            wrist_point,
            effective_bag_center
        )


        if (
            wrist_distance
            <=
            WRIST_BAG_DISTANCE
        ):

            wrist_near_bag = True


    # =========================================
    # RELEASE CHECK
    # =========================================

    wrist_released = False


    if (
        wrist_point is not None
        and
        effective_bag_center is not None
    ):

        if (
            wrist_distance
            >
            RELEASE_DISTANCE
        ):

            wrist_released = True


    # =========================================
    # BAG MOVEMENT CHECK
    # =========================================

    bag_moved_from_start = False

    movement = 0


    if (
        effective_bag_center is not None
        and
        initial_bag_center is not None
    ):

        movement = distance(
            effective_bag_center,
            initial_bag_center
        )


        if (
            movement
            >
            BAG_MOVEMENT_THRESHOLD
        ):

            bag_moved_from_start = True


    # =========================================
    # STATE MACHINE
    # =========================================

    if task_completed:

        state = "PLACED"

    elif state == "PLACED":

        if (
            person_found
            and
            bag_found
            and
            (
                wrist_inside
                or
                wrist_near_bag
            )
        ):

            wrist_inside_count += 1

        else:

            wrist_inside_count = 0

        if wrist_inside_count >= ENTER_FRAMES:

            state = "PICKING"

            wrist_inside_count = 0
            bag_moving_count = 0


    elif state == "PICKING":

        picking_detected = False

        if (
            bag_found
            and
            bag_moved_from_start
        ):

            picking_detected = True

        elif (
            person_found
            and
            not bag_found
            and
            wrist_point is not None
            and
            wrist_near_bag
        ):

            picking_detected = True

        elif (
            person_found
            and
            wrist_near_bag
            and
            not bag_inside
        ):

            picking_detected = True


        if picking_detected:

            bag_moving_count += 1

        else:

            bag_moving_count = 0


        if bag_moving_count >= PICK_FRAMES:

            state = "PICKED"

            bag_moving_count = 0
            leave_count = 0


    elif state == "PICKED":

        if not person_found:

            leave_count += 1

        else:

            leave_count = 0


        if leave_count >= LEAVE_FRAMES:

            state = "PERSON LEAVES"

            leave_count = 0
            return_count = 0


    elif state == "PERSON LEAVES":

        placing_interaction = (

            person_found
            and
            bag_found
            and
            bag_inside
            and
            (
                wrist_inside
                or
                wrist_near_bag
            )
        )


        if placing_interaction:

            return_count += 1

        else:

            return_count = 0


        if return_count >= RETURN_FRAMES:

            state = "PLACING"

            return_count = 0
            release_count = 0


    elif state == "PLACING":

        bag_stable = False

        if (
            bag_found
            and
            previous_bag_center is not None
            and
            bag_center is not None
        ):

            recent_movement = distance(
                bag_center,
                previous_bag_center
            )

            bag_stable = recent_movement < 15


        current_wrist = None

        if (
            wrists
            and
            effective_bag_center is not None
        ):

            current_wrist = closest_wrist(
                wrists,
                effective_bag_center
            )


        actual_wrist_released = False

        if current_wrist is None:

            actual_wrist_released = True

        elif effective_bag_center is not None:

            actual_wrist_distance = distance(
                current_wrist,
                effective_bag_center
            )

            if actual_wrist_distance > RELEASE_DISTANCE:

                actual_wrist_released = True


        released = (
            bag_found
            and
            bag_inside
            and
            (
                actual_wrist_released
                or
                bag_stable
            )
        )


        if released:

            release_count += 1

        else:

            release_count = 0


        if release_count >= RELEASE_FRAMES:

            state = "PLACED"
            task_completed = True

            release_count = 0
            wrist_inside_count = 0
            bag_moving_count = 0
            return_count = 0


    # =========================================
    # HAND DISPLAY
    # =========================================

    if (
        state == "PICKED"
        or
        state == "PERSON LEAVES"
        or
        state == "PLACED"
    ):

        display_interacting = False

    else:

        display_interacting = wrist_near_bag


    # =========================================
    # DRAW PERSON
    # =========================================

    if person_center is not None:

        cv2.circle(
            frame,
            person_center,
            6,
            (0, 255, 0),
            -1
        )


        cv2.putText(
            frame,
            "PERSON",
            (
                person_center[0],
                max(
                    20,
                    person_center[1] - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


        if person_id is not None:

            cv2.putText(
                frame,
                "Person ID: 1",
                (
                    person_center[0],
                    person_center[1] + 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )


    # =========================================
    # DRAW WRIST
    # =========================================

    if wrist_point is not None:

        cv2.circle(
            frame,
            wrist_point,
            8,
            (0, 255, 255),
            -1
        )


        cv2.putText(
            frame,
            "WRIST",
            (
                wrist_point[0] + 10,
                wrist_point[1]
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )


    # =========================================
    # DRAW ROI
    # =========================================

    cv2.rectangle(
        frame,
        (ROI_X1, ROI_Y1),
        (ROI_X2, ROI_Y2),
        (255, 255, 0),
        3
    )


    cv2.putText(
        frame,
        "BACKPACK ROI",
        (
            ROI_X1,
            max(
                20,
                ROI_Y1 - 10
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2
    )


    # =========================================
    # STATE DISPLAY
    # =========================================

    cv2.rectangle(
        frame,
        (10, 10),
        (530, 170),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        f"BAG STATE: {state}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    if display_interacting:

        cv2.putText(
            frame,
            "HAND: INTERACTING",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "HAND: NOT INTERACTING",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


    if np.isfinite(wrist_distance):

        wrist_distance_display = int(
            wrist_distance
        )

    else:

        wrist_distance_display = 0


    cv2.putText(
        frame,
        f"Bag movement: {int(movement)} px",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        f"Wrist-Bag: {wrist_distance_display} px",
        (20, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        f"ROI overlap: {roi_overlap * 100:.1f}%",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )


    # =========================================
    # SAVE OUTPUT
    # =========================================

    writer.write(frame)


    cv2.imshow(
        "Task 1 - Backpack Pick and Place",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    if key == 27:

        break


cap.release()

writer.release()

cv2.destroyAllWindows()


print()
print("========================================")
print("TASK 1 COMPLETED")
print("========================================")
print()
print("Output video:")
print(OUTPUT_PATH)
print()
print("Final state:")
print(state)
print()
print("========================================")