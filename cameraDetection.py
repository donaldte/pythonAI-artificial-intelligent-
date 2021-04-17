import cv2
from random import randrange

# load some pre-trained data on face frontal from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Choose an image to detect faces
# img = cv2.imread('fc.jpg')

# to capture video from webcam
webcam = cv2.VideoCapture(0)
# Iterate forever over frames
while True:
    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # Must convert to grayscale
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
    for (X, Y, W, H) in face_coordinates:
        home = cv2.rectangle(frame, (X, Y), (X + W, Y + H), (randrange(256), randrange(255), randrange(256)), 10)
        cv2.imshow('converted image', home)
        cv2.waitKey(1)
        # Stop if key Q is press
        #if Key == 83 or Key == 113:
         #   break

    # Release the videoCapture object
            #webcam.release()

    #print("code completed")
