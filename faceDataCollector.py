import cv2
import numpy as np
name=input("enter name")
cam=cv2.VideoCapture(0)
reco=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
count=0
facedata=[]
while True:
    ret,frame=cam.read()
    if ret==False:
        continue
    faces=reco.detectMultiScale(frame,1.3,5)
    faces=sorted(faces,key=lambda f:f[2]*f[3])
    for x,y,w,h in faces[-1:]:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        facesec=frame[y-10:y+h+10,x-10:x+w+10]
        facesec=cv2.resize(facesec,(100,100))
        if(count%10==0):
            facedata.append(facesec)
            print(len(facedata))
        count+=1
        
    cv2.imshow("video",frame)
    key=cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
facedata=np.array(facedata)
facedata=facedata.reshape((facedata.shape[0],-1))
np.save("./data/"+name+".npy",facedata)
print(facedata.shape)
cam.release()
cv2.destroyAllWindows()