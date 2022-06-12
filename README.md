# INFORMATION RETRIEVAL SYSTEM FROM IMAGES DATABASE
## Face-recognition-attendance-system
### Face recognition made using python

Libraries required: → cmake → dlib → numpy → opencv → face recognition → os 

⦁ Methodology :

The proposed system face recognition-based attendance system can be divided into four main steps .The steps and their functions are defined as follows.

⦁ Step 1: Finding all the Faces. ⦁ Step 2: Posing and Projecting Faces. ⦁ Step 3: Encoding Faces. ⦁ Step 4: Finding the person’s name from the encoding.

⦁ Step 1: Finding all the Faces:

⦁ The first step in our pipeline is face detection.  ⦁  We’re going to use a method invented in 2005 called Histogram of Oriented Gradients — or just HOG for short to detect the faces. ⦁ To find faces in an image, we’ll start by making our image black and white because we don’t need color data to find faces. ⦁ To find faces in this HOG image, all we have to do is find the part of our image that looks the most similar to a known HOG pattern that was extracted from a bunch of other training faces.

⦁ Step 2: Posing and Projecting Faces:

⦁ Now, we have detected the faces in our image. But now we have to deal with the problem that faces turned different direction look totally different to a computer. ⦁ To account for this, we will try to warp each picture so that the eyes and lips are always in the sample place in the image.  ⦁ This will make it a lot easier for us to compare faces in the next steps. ⦁ To do this , we can use an algorithm called “face landmark estimation”. ⦁ The basic idea is we will come up with 68 specific points(called landmarks) that exist  on every face - the top of the chin,the outside edge of each eye,the inner edge of each eye brow,etc. ⦁ Then we can train a machine learning algorithm to be able to find these 68 specific points on any face. ⦁ Now that we know where the mouth and eyes are, we will simply rotate, scale and shear the image so that the eyes and mouth are centered as best as possible. ⦁ Now no matter how the face is turned we are able to center the eyes and mouth are in roughly the same position in the image.

⦁ Step 3: Encoding faces:

⦁ This is actually telling faces apart. ⦁ In order to do this we extract a few basic measurements from each face. Then we could measure our unknown face the same way and find the known face with the closest measurements.  ⦁ For example, we might measure the size of each ear, the spacing between the eyes, the length of the nose, etc. ⦁ The solution is to train a Deep Convolutional Neural Network .  ⦁ But instead of training the network to recognize pictures objects like we did last time, we are going to train it to generate 128 measurements for each face.

The training process works by looking at 3 face images at a time: ⦁ Load a training face image of a known person. ⦁ Load another picture of the same known person. ⦁ Load a picture of a totally different person. ⦁ Then the algorithm looks at the measurements it is currently generating for each of those three images. It then tweaks the neural network slightly so that it makes sure the measurements it generates for #1 and #2 are slightly closer while making sure the measurements for #2 and #3 are slightly further apart. ⦁ After repeating this step millions of times for millions of images of thousand of different people,the neural network learns to reliably generate 128 measurements for each person.Any ten different pictures of the same person should give you roughly the same measurement.

⦁ Step 4: Finding the person’s name from the encoding:

⦁ All we have to do is find the person in our database of known people who has the closest measurements to our test image. ⦁ We’ll use a simple linear SVM classifier. ⦁ We have to train a classifier that can take in the measurements from a new test image and tells which known person is the closest match. ⦁  Running this classifier takes milliseconds. The result of the classifier is the name of the person. ⦁ Finally  we store this name into an excel sheet.
