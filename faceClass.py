import cv2
import numpy as np
import os

from numpy.core.records import array
classid=0
face_data=[]
face_label=[]
names={}
for f in os.listdir("./data/"):
    names[classid]=f[:-4]
    if f.endswith('.npy'):
        data=np.load("./data/"+f)
        face_data.append(data)
        
        label=classid*np.ones((data.shape[0],))
        face_label.append(label)
        classid+=1

face_data=np.concatenate(face_data,axis=0)
face_label=np.concatenate(face_label,axis=0).reshape(-1,1)
print(face_data.shape," ",face_label.shape)
dataset=np.concatenate((face_data,face_label),axis=1)
print(dataset.shape)

def predict(val,k=15):
    X=dataset[:,:-1]
    Y=dataset[:,-1]
    pt=np.array(val)
    a=(X-pt)**2
    dis=(a[:][:,0]+a[:][:,1])**0.5
    idx=dis.argsort()[:k]
    y_val=Y[idx]
    clas,fre=np.unique(y_val,return_counts=True)
    print("accuracy=",max(fre)/k)
    return clas[fre.argmax()],max(fre)/k

cam=cv2.VideoCapture(0)
reco=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    rec,frame=cam.read(0)
    if rec==False:
        continue
    faces=reco.detectMultiScale(frame,1.3,5)
    for x,y,w,h in faces:
        face_sec=frame[y-10:y+h+10,x-10:x+w+10]
        face_sec=cv2.resize(face_sec,(100,100))
        result,acc=predict(face_sec.flatten())
        if acc>0.6:
            cv2.putText(frame,names[result],(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("video",frame)
    key=cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

