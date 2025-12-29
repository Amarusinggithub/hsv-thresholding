import cv2
import numpy as np
import imutils


KNOWN_DISTANCE = 20.0  
KNOWN_WIDTH = 11.0  



def find_marker(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) == 0:
        return None

    c = max(cnts, key=cv2.contourArea)
    return cv2.minAreaRect(c)


def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth


def calculate_focal_length(measured_distance, real_width, width_in_pixels):
    return (width_in_pixels * measured_distance) / real_width


cap = cv2.VideoCapture(0)

focalLength = 0
is_calibrated = False

while True:
    ret, image = cap.read()
    if not ret:
        break

    marker = find_marker(image)

    
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 120), (0, 0, 0), -1)
    alpha = 0.6 
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    if marker is not None:
        # Get object width in pixels
        box = cv2.boxPoints(marker) 
        box = np.int64(box)
        pixel_width = max(marker[1][0], marker[1][1])

        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

       
        if not is_calibrated:
         
            cv2.putText(
                image,
                "CALIBRATION MODE",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                image,
                f"Hold object at {KNOWN_DISTANCE} in",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                image,
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
                image,
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
                image,
                counter_text,
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                text_color,
                3,
            )

    else:
        
        cv2.putText(
            image,
            "SEARCHING...",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )

    cv2.imshow("Live Distance Counter", image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
