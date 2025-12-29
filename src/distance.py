import cv2
import numpy as np
import imutils


KNOWN_DISTANCE = 20.0  
KNOWN_WIDTH = 11.0  


lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

focalLength = 0
is_calibrated = False

def find_marker(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    cv2.imshow("Mask", mask)

    contours= cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if len(contours) == 0:
        return None

    contours = max(contours, key=cv2.contourArea)
    return cv2.minAreaRect(contours)


def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth


def calculate_focal_length(measured_distance, real_width, width_in_pixels):
    return (width_in_pixels * measured_distance) / real_width


cap = cv2.VideoCapture(0)


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
        # Get object width in pixels
        box = cv2.boxPoints(marker) 
        box = np.int64(box)
        pixel_width = max(marker[1][0], marker[1][1])
        cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)

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
                "Press 'c' to set",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                focalLength = calculate_focal_length(
                    KNOWN_DISTANCE, KNOWN_WIDTH, pixel_width
                )
                is_calibrated = True
                print(f"Calibrated! Focal Length: {focalLength}")

        else:
            # this Calculate Distance
            inches = distance_to_camera(KNOWN_WIDTH, focalLength, pixel_width)

            cv2.putText(
                frame,
                "DISTANCE LIVE:",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )

            text_color = (0, 255, 0) if inches < 30 else (0, 0, 255)

            counter_text = f"{inches:.2f} in"
            cv2.putText(
                frame,
                counter_text,
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                text_color,
                3,
            )

    else:

        cv2.putText(
            frame,
            "SEARCHING...",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )

    cv2.imshow("Live Distance Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
