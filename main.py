import cv2

# load some pre-trained data on face frontal from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Choose an image to detect faces
img = cv2.imread('fc.jpg')

# Must convert to grayscale
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
# Draw rectangle around the faces
(X, Y, W, H) = face_coordinates[0]
home = cv2.rectangle(img, (X, Y), (X+W, Y+H), (0, 255, 0), 2)

cv2.imshow('convert image', home)

cv2.waitKey()
