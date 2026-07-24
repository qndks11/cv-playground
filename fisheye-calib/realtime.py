import cv2
from undistort import undistort

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    undistorted_frame = undistort(frame, balance=1, dim2=(1920, 1080))
    cv2.imshow("Frame", undistorted_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break