import cv2
import numpy as np

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

cap = cv2.VideoCapture(0)


while True:
    
    ret, frame = cap.read()
  

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

   
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) !=0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour) 

        if w > 20 and h > 20:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 1), 2)

            cv2.putText(
                frame,
                f"Duck (x: {x}, y: {y})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

    cv2.imshow("Camera Feed", frame)
    
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
