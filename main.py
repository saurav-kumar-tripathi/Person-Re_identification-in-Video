import cv2
import os
cascPath = os.path.dirname(cv2.__file__) + \
    "/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_DUPLEX
text = 'Saurav Kumar Tripathi'
while True:
    ret, frames = video_capture.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.putText(frames, text, (x+w, y+h), font, 255, color=(255, 0, 0))
    cv2.imshow('video', frames)
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break
video_capture.release()
cv2.destroyAllWindows()
