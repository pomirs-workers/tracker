import cv2 as cv
import imutils

cap = cv.VideoCapture(0)

MIN_AREA = 1000

def get_frame(vid):
    r, frame = vid.read()
    gray_blured = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_blured = cv.GaussianBlur(gray_blured, (21, 21), 0)
    return frame, gray_blured


frameOld = None
_, frameNow = get_frame(cap)

while True:
    frameOld = frameNow
    frameColor, frameNow = get_frame(cap)
    delta = cv.absdiff(frameOld, frameNow)
    ret, thresh = cv.threshold(delta, 25, 255, cv.THRESH_BINARY)
    thresh = cv.dilate(thresh, None, iterations=2)
    contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    for c in contours:
        if cv.contourArea(c) < MIN_AREA:
            continue
        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(frameColor, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(x, y)
    cv.imshow("delta", frameColor)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()

