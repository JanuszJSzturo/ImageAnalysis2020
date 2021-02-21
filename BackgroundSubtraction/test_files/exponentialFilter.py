import numpy as np
import cv2

cap = cv2.VideoCapture('project.avi')

kernel = np.ones((2,2),np.uint8)

init_frames = 20
alpha = 0.05 #learning rate [0,1]

background = np.zeros(shape=(240,320))
for i in range(init_frames):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    background = background + frame

background = background/init_frames
background = background.astype(np.uint8)

while(1):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mask_frame = np.abs(frame - background)

    ret, thresh1 = cv2.threshold(mask_frame, 100, 255, cv2.THRESH_BINARY)
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    background = ((1-alpha)*background+alpha*frame).astype(np.uint8)

    cv2.imshow('frame', thresh1)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()