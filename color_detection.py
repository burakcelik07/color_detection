import cv2
import numpy as np


def nothing(x):
    pass


cap = cv2.VideoCapture(0)

cv2.namedWindow("trackbar")

cv2.createTrackbar("upper_blue", "trackbar", 0, 255, nothing)
cv2.createTrackbar("upper_green", "trackbar", 0, 255, nothing)
cv2.createTrackbar("upper_red", "trackbar", 0, 255, nothing)
cv2.createTrackbar("lower_blue", "trackbar", 0, 255, nothing)
cv2.createTrackbar("lower_green", "trackbar", 0, 255, nothing)
cv2.createTrackbar("lower_red", "trackbar", 0, 255, nothing)

while True:
    upper_blue = cv2.getTrackbarPos("upper_blue", "trackbar")
    upper_green = cv2.getTrackbarPos("upper_green", "trackbar")
    upper_red = cv2.getTrackbarPos("upper_red", "trackbar")
    lower_blue = cv2.getTrackbarPos("lower_blue", "trackbar")
    lower_green = cv2.getTrackbarPos("lower_green", "trackbar")
    lower_red = cv2.getTrackbarPos("lower_red", "trackbar")

    lower_color = np.array([lower_blue, lower_green, lower_red], dtype=np.uint8)
    upper_color = np.array([upper_blue, upper_green, upper_red], dtype=np.uint8)

    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    blur = cv2.medianBlur(frame, 15)

    mask = cv2.inRange(blur, lower_color, upper_color)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)

        x = approx.ravel()[0]
        y = approx.ravel()[1]
        if area > 400:
            if len(approx) > 5:
                cv2.putText(frame, "Red Circle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), thickness=2)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()