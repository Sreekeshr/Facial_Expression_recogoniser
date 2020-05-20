import cv2
from my_model import FacialExpressionRecogonizer
import numpy as np
model = FacialExpressionRecogonizer("/home/sreekesh/python/VS CODE/Face_recognition/Facial_Expression_Recognition/one/new_c_model.json","/home/sreekesh/python/VS CODE/Face_recognition/Facial_Expression_Recognition/one/new_c_model.h5")


face_cas = cv2.CascadeClassifier("/home/sreekesh/harcascade_face.xml")
cap = cv2.VideoCapture(0)

while(True):
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray,1.4,5)

    for (x,y,w,h) in faces :
        fc = gray[y:y+h,x:x+w]
        roi = cv2.resize(fc,(48,48))
        pre = model.predict_emotion(roi[np.newaxis,:,:,np.newaxis])
        cv2.putText(frame,pre,(x+w,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('my_face',frame)
    if cv2.waitKey(1) & 0xff  == ord('q') :
        break

cap.release()
cv2.destroyAllWindows()