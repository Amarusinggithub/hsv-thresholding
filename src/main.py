import cv2
import numpy as np
import imutils
import argparse
from ultralytics import YOLO


KNOWN_DISTANCE = 44.0 
KNOWN_WIDTH = 18.0  

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

focalLength = 0
is_calibrated = False


def find_marker(frame):
    """Find the colored marker in the frame and return its bounding rectangle."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    cv2.imshow("Mask", mask)

    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if len(contours) == 0:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest_contour) < 500:
        return None

    return cv2.minAreaRect(largest_contour)


def distance_to_camera(knownWidth, focalLength, perWidth):
    if perWidth == 0:
        return 0
    return (knownWidth * focalLength) / perWidth


def calculate_focal_length(measured_distance, real_width, width_in_pixels):
    if width_in_pixels == 0:
        return 0
    return (width_in_pixels * measured_distance) / real_width


def detect_from_webcam(model_path: str = "yolov8n.pt", confidence: float = 0.5):
    global focalLength, is_calibrated

    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Press 'q' to quit")
    print(
        f"Hold your object (width: {KNOWN_WIDTH} in) at {KNOWN_DISTANCE} inches and press 'c' to calibrate"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        marker = find_marker(frame)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (350, 120), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        if marker is not None:
            box = cv2.boxPoints(marker)
            box = np.int64(box)
            cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)

            pixel_width = max(marker[1][0], marker[1][1])

            if not is_calibrated:
                cv2.putText(
                    frame,
                    "CALIBRATION MODE",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Hold object at {KNOWN_DISTANCE} in",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    frame,
                    f"Pixel width: {pixel_width:.0f}px",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    frame,
                    "Press 'c' to calibrate",
                    (180, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
            else:
                inches = distance_to_camera(KNOWN_WIDTH, focalLength, pixel_width)

                cv2.putText(
                    frame,
                    "DISTANCE:",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                )

                text_color = (
                    (0, 255, 0)
                    if inches < 30
                    else (0, 165, 255) if inches < 50 else (0, 0, 255)
                )

                cv2.putText(
                    frame,
                    f"{inches:.1f} in",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    text_color,
                    3,
                )
                cv2.putText(
                    frame,
                    f"({inches * 2.54:.1f} cm)",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    text_color,
                    2,
                )
        else:
            if not is_calibrated:
                cv2.putText(
                    frame,
                    "CALIBRATION MODE",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
            cv2.putText(
                frame,
                "SEARCHING FOR OBJECT...",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        results = model(frame, conf=confidence, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow("Distance Measurement", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c") and marker is not None and not is_calibrated:
            pixel_width = max(marker[1][0], marker[1][1])
            focalLength = calculate_focal_length(
                KNOWN_DISTANCE, KNOWN_WIDTH, pixel_width
            )
            is_calibrated = True
            print(f"Calibrated! Focal Length: {focalLength:.2f}")
        elif key == ord("r"):
            is_calibrated = False
            focalLength = 0
            print("Calibration reset")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Distance Measurement with YOLO")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model path (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="yellow",
        choices=["yellow", "red", "green", "blue"],
        help="Color of object to track (default: yellow)",
    )

    args = parser.parse_args()

    # Set color range based on argument
    global lower_yellow, upper_yellow
    color_ranges = {
        "yellow": ([20, 100, 100], [30, 255, 255]),
        "red": ([0, 100, 100], [10, 255, 255]),
        "green": ([40, 100, 100], [80, 255, 255]),
        "blue": ([100, 100, 100], [130, 255, 255]),
    }
    lower_yellow = np.array(color_ranges[args.color][0])
    upper_yellow = np.array(color_ranges[args.color][1])

    print(f"Tracking {args.color} objects")
    detect_from_webcam(args.model, args.confidence)


if __name__ == "__main__":
    main()
