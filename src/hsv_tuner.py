import cv2
import numpy as np


def nothing(x):
    pass


cap = cv2.VideoCapture(0)


cv2.namedWindow("Trackbars")
cv2.createTrackbar("H_min", "Trackbars", 20, 179, nothing) 
cv2.createTrackbar("H_max", "Trackbars", 30, 179, nothing)
cv2.createTrackbar("S_min", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("S_max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V_min", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("V_max", "Trackbars", 255, 255, nothing)



while True:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("H_min", "Trackbars")
    h_max = cv2.getTrackbarPos("H_max", "Trackbars")
    s_min = cv2.getTrackbarPos("S_min", "Trackbars")
    s_max = cv2.getTrackbarPos("S_max", "Trackbars")
    v_min = cv2.getTrackbarPos("V_min", "Trackbars")
    v_max = cv2.getTrackbarPos("V_max", "Trackbars")

    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Camera Feed", frame)
    cv2.imshow("Mask", mask)
    #cv2.imshow("Result", result)

    if cv2.waitKey(1) == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
