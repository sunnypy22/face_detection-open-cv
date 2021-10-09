import cv2
import os

# load the cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# Video Capturing
video_capture = cv2.VideoCapture(0)
while True:
    # read every Frame
    _, img = video_capture.read()

    # Convert image to grayscales image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect The Face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                                          , flags=cv2.CASCADE_SCALE_IMAGE)
    # Draw Rectangle arround the face
    for (x, y, w, h) in faces:
        data = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imwrite("E:/Interview/AHM_test/face_detection/test.png", data)
        cv2.putText(img, "Face Detected", (30, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (255, 0, 0), 2)

    cv2.imshow("img", img)
    # escape ley to exit
    k = cv2.waitKey(30) & 0xff

    if k == 27:
        break

video_capture.release()
cv2.destroyAllWindows()
