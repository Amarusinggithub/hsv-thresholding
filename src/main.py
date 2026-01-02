import cv2
import numpy as np
import imutils
import argparse
from ultralytics import YOLO


KNOWN_DISTANCE = 44.0 
KNOWN_WIDTH = 18.0 


focalLength = 0
is_calibrated = False


lower_color = np.array([20, 100, 100])
upper_color = np.array([30, 255, 255])


def find_marker(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if len(contours) == 0:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest_contour) < 500:
        return None

    return cv2.boundingRect(largest_contour)


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

    print("Loading YOLO model...")
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Press 'q' to quit")
    print(f"Setup: Hold object ({KNOWN_WIDTH} in wide) at {KNOWN_DISTANCE} in away.")
    print("Press 'c' to CALIBRATE when ready.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=confidence, verbose=False)
        annotated_frame = results[0].plot()

        marker = find_marker(frame)

        if marker is not None:
            x, y, w, h = marker

            # Draw the box around the colored object
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            pixel_width = w

            if not is_calibrated:
                cv2.putText(
                    annotated_frame,
                    "CALIBRATION MODE",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    annotated_frame,
                    f"Width: {pixel_width}px",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    annotated_frame,
                    "Press 'c' to set",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

            else:
                inches = distance_to_camera(KNOWN_WIDTH, focalLength, pixel_width)

                # Dynamic text color
                text_color = (0, 255, 0)  
                if inches > 50:
                    text_color = (0, 0, 255)  
                elif inches > 30:
                    text_color = (0, 165, 255)  

                label_text = f"{inches:.1f} in"

                # Calculate text size for the background box
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                )

                # Logic to make text float above object (or below if at top of screen)
                if y - 30 > 0:
                    text_x = x
                    text_y = y - 10
                else:
                    text_x = x
                    text_y = y + h + 25

                # Draw black background rectangle for text
                cv2.rectangle(
                    annotated_frame,
                    (text_x, text_y - text_h - 5),
                    (text_x + text_w, text_y + 5),
                    (0, 0, 0),
                    -1,
                )

                # Draw the distance text
                cv2.putText(
                    annotated_frame,
                    label_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    text_color,
                    2,
                )

        else:
            if not is_calibrated:
                cv2.putText(
                    annotated_frame,
                    "Looking for color marker...",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        cv2.imshow("Distance Measurement", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c") and marker is not None and not is_calibrated:
            # Calibrate using the current width
            pixel_width = marker[2]
            focalLength = calculate_focal_length(
                KNOWN_DISTANCE, KNOWN_WIDTH, pixel_width
            )
            is_calibrated = True
            print(f"Calibration Complete! Focal Length: {focalLength:.2f}")
        elif key == ord("r"):
            # Reset calibration
            is_calibrated = False
            focalLength = 0
            print("Calibration reset")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Distance Measurement with YOLO")
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt", help="YOLO model path"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="Detection confidence"
    )
    parser.add_argument(
        "--color",
        type=str,
        default="yellow",
        choices=["yellow", "red", "green", "blue"],
        help="Color to track",
    )

    args = parser.parse_args()

    # Set global color range
    global lower_color, upper_color
    color_ranges = {
        "yellow": ([20, 100, 100], [30, 255, 255]),
        "red": ([0, 100, 100], [10, 255, 255]),
        "green": ([40, 100, 100], [80, 255, 255]),
        "blue": ([100, 100, 100], [130, 255, 255]),
    }

    lower_color = np.array(color_ranges[args.color][0])
    upper_color = np.array(color_ranges[args.color][1])

    print(f"Tracking {args.color} object...")
    detect_from_webcam(args.model, args.confidence)


if __name__ == "__main__":
    main()
