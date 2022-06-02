# import cv2
# import numpy as np
# import face_recognition
# import os
# from datetime import datetime

# path = 'ImagesAttendance'
# images = []
# classNames = []
# myList = os.listdir(path)
# print(myList)

# for cl in myList:
#     curImg = cv2.imread(f'{path}/{cl}')
#     images.append(curImg)
#     classNames.append(os.path.splitext(cl)[0])
# print(classNames)


# def findEncodings(images):
#     encodeList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#     return encodeList


# def markAttendance(name):
#     with open('Attendance.csv', 'r+') as f:
#         myDataList = f.readlines()
#         nameList = []
#         for line in myDataList:
#             entry = line.split(',')
#             nameList.append(entry[0])
#         if name not in nameList:
#             now = datetime.now()
#             dtString = now.strftime('%H:%M:%S')
#             f.writelines(f'\n{name},{dtString}')


# encodeListKnown = findEncodings(images)
# print('Encoding Complete')

# cap = cv2.VideoCapture(0)

# while True:
#     success, img = cap.read()
#     # img = captureScreen()
#     imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

#     facesCurFrame = face_recognition.face_locations(imgS)
#     encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

#     for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
#         matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
#         faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
#         matchIndex = np.argmin(faceDis)


#         if matches[matchIndex]:
#             name = classNames[matchIndex].upper()
#             y1, x2, y2, x1 = faceLoc
#             y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
#             cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
#             markAttendance(name)

#         cv2.imshow('Webcam', img)
#         cv2.waitKey(1)
        
        
        
#         Face-recognition-attendance-system
# Face recognition made using python

# Libraries required: -> cmake -> dlib -> numpy -> opencv -> face recognition -> os ⦁ Methodology :

# The proposed system face recognition-based attendance system can be divided into four main steps .The steps and their functions are defined as follows.

# ⦁ Step 1: Finding all the Faces. ⦁ Step 2: Posing and Projecting Faces. ⦁ Step 3: Encoding Faces. ⦁ Step 4: Finding the person’s name from the encoding.

# ⦁ Step 1: Finding all the Faces:

# ⦁ The first step in our pipeline is face detection.  ⦁  We’re going to use a method invented in 2005 called Histogram of Oriented Gradients — or just HOG for short to detect the faces. ⦁ To find faces in an image, we’ll start by making our image black and white because we don’t need color data to find faces. ⦁ To find faces in this HOG image, all we have to do is find the part of our image that looks the most similar to a known HOG pattern that was extracted from a bunch of other training faces.

# ⦁ Step 2: Posing and Projecting Faces:

# ⦁ Now, we have detected the faces in our image. But now we have to deal with the problem that faces turned different direction look totally different to a computer. ⦁ To account for this, we will try to warp each picture so that the eyes and lips are always in the sample place in the image.  ⦁ This will make it a lot easier for us to compare faces in the next steps. ⦁ To do this , we can use an algorithm called “face landmark estimation”. ⦁ The basic idea is we will come up with 68 specific points(called landmarks) that exist  on every face - the top of the chin,the outside edge of each eye,the inner edge of each eye brow,etc. ⦁ Then we can train a machine learning algorithm to be able to find these 68 specific points on any face. ⦁ Now that we know where the mouth and eyes are, we will simply rotate, scale and shear the image so that the eyes and mouth are centered as best as possible. ⦁ Now no matter how the face is turned we are able to center the eyes and mouth are in roughly the same position in the image.

# ⦁ Step 3: Encoding faces:

# ⦁ This is actually telling faces apart. ⦁ In order to do this we extract a few basic measurements from each face. Then we could measure our unknown face the same way and find the known face with the closest measurements.  ⦁ For example, we might measure the size of each ear, the spacing between the eyes, the length of the nose, etc. ⦁ The solution is to train a Deep Convolutional Neural Network .  ⦁ But instead of training the network to recognize pictures objects like we did last time, we are going to train it to generate 128 measurements for each face.

# The training process works by looking at 3 face images at a time: ⦁ Load a training face image of a known person. ⦁ Load another picture of the same known person. ⦁ Load a picture of a totally different person. ⦁ Then the algorithm looks at the measurements it is currently generating for each of those three images. It then tweaks the neural network slightly so that it makes sure the measurements it generates for #1 and #2 are slightly closer while making sure the measurements for #2 and #3 are slightly further apart. ⦁ After repeating this step millions of times for millions of images of thousand of different people,the neural network learns to reliably generate 128 measurements for each person.Any ten different pictures of the same person should give you roughly the same measurement.

# ⦁ Step 4: Finding the person’s name from the encoding:

# ⦁ All we have to do is find the person in our database of known people who has the closest measurements to our test image. ⦁ We’ll use a simple linear SVM classifier. ⦁ We have to train a classifier that can take in the measurements from a new test image and tells which known person is the closest match. ⦁  Running this classifier takes milliseconds. The result of the classifier is the name of the person. ⦁ Finally  we store this name into an excel sheet.