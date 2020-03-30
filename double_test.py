import cv2 as cv
import imutils

cam_xz = cv.VideoCapture(1)
cam_yz = cv.VideoCapture(2)

while True:
    r, frame_xz = cam_xz.read()
    r, frame_yz = cam_yz.read()
    cv.imshow("xz", frame_xz)
    cv.imshow("yz", frame_yz)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()

