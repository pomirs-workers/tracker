import functions as fn
import cv2 as cv
import imutils

cap_xz = cv.VideoCapture(0)     # Stream from Cam №1 (X and Y coordinates)
cap_yz = cv.VideoCapture(1)     # Stream from Cam №2 (X and Z coordinates).
previous_frame_xz = None
previous_frame_yz = None        # Frame to compare with

max_X, max_Y, max_Z = 200, 200, 500

blind_spot_xz = 0.1   # blind spot the 1st cam in shares
blind_spot_yz = 0.1   # blind spot the 1st cam in shares

while True:
    _, frame_xz = cap_xz.read()
    _, frame_yz = cap_yz.read()

    frame_xz = cv.cvtColor(frame_xz[:, 0+int(frame_xz.shape[1]*blind_spot_xz):int(frame_xz.shape[1]*(1-blind_spot_xz))],
                           cv.COLOR_BGR2GRAY)
    frame_yz = cv.cvtColor(frame_yz[:, 0+int(frame_yz.shape[1]*blind_spot_yz):int(frame_yz.shape[1]*(1-blind_spot_yz))],
                           cv.COLOR_BGR2GRAY)

    frame_xz = cv.GaussianBlur(frame_xz, (5, 5), 0)
    frame_yz = cv.GaussianBlur(frame_yz, (5, 5), 0)

    if (previous_frame_xz is None) or (previous_frame_yz is None):
        previous_frame_xz = frame_xz.copy()
        previous_frame_yz = frame_yz.copy()
        continue


    frames_diff_xz = cv.absdiff(frame_xz, previous_frame_xz)
    frames_diff_yz = cv.absdiff(frame_yz, previous_frame_yz)

    thresh_frame_xz = cv.threshold(frames_diff_xz, 47, 230 ,cv.THRESH_BINARY)
    thresh_frame_yz = cv.threshold(frames_diff_yz, 47, 230, cv.THRESH_BINARY)

    thresh_frame_xz = cv.dilate(thresh_frame_xz, iterations=3)
    thresh_frame_yz = cv.dilate(thresh_frame_yz, iterations=3)

    cnts_xz = cv.findContours(thresh_frame_xz.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts_yz = cv.findContours(thresh_frame_yz.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cnts_xz = imutils.grab_contours(cnts_xz)
    cnts_yz = imutils.grab_contours(cnts_yz)

    for c in cnts_xz:
        if cv.contourArea(c)<500
            continue


        (x, y, w, h) = cv.boundingRect(c)