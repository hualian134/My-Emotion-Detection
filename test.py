import cv2
import keras
import tensorflow

from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.utils import img_to_array
from keras._tf_keras.keras.preprocessing import image
import numpy as np 
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#print(keras.__version__)
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classifier = load_model(r'Emotion_little_vgg1.h5')
class_labels = ['Angry','Happy','Neutral','Sad','Surprise','Disgust','Fear']

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(r'video.mp4')
while (cap.isOpened()):

    ret, frame = cap.read()
    #labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    # rect,face,image = face_detector(frame)
        if np.sum([roi_gray]) > 0:
            roi = img_to_array(roi_gray)
            roi = roi.astype('float')/255.0
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class
            predictions = classifier.predict(roi)[0]
            label=class_labels[predictions.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()