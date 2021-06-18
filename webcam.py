import cv2
cam=cv2.VideoCapture(0)
while True:
    ret,frame=cam.read()
    if(ret==False):
        continue
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    cv2.imshow('video stream',frame)
    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()



