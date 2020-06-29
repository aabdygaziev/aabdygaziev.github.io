---
layout: post
title: Face Detection with Python & OpenCV using web-cam
---
***


![png](/images/face_id/download.png)

OpenCV is very powerfull library designed to solve computer vision problems. OpenCV-Python makes use of Numpy, which is a highly optimized library for numerical operations with a MATLAB-style syntax. All the OpenCV array structures are converted to and from Numpy arrays. This also makes it easier to integrate with other libraries that use Numpy such as SciPy and Matplotlib (OpenCV documentation).

To be able to detect faces real-time we will use **HAAR classifier**. It's a machine learning based algorithm to identify object in an image or video. You can learn more about Haar and computer vision on <a href='http://www.willberger.org/cascade-haar-explained/'>Will Bergers website</a>. Also on <a href='https://en.wikipedia.org/wiki/Haar-like_feature#:~:text=Haar%2Dlike%20features%20are%20digital,first%20real%2Dtime%20face%20detector.'>wikipedia</a>.

Before starting, you need to download <a href='https://github.com/opencv/opencv/tree/master/data/haarcascades'>haarcascade_frontalface_default.xml</a> file on github.

First, we need to import Numpy and OpenCV.


```python
import numpy as np
import cv2
```

Next:
* We difine our classifier at**face_cascade**.
* **cap = cv2.VideoCapture(0)** returns video from your fisrt webcamera
* **cap.set** method is optional if you want to resize your frame.


```python
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 250)
cap.set(cv2.CAP_PROP_FPS, 25)
```




    True



**while True** initiates infinite loop. 
* ret, img = cap.read() - ret is boolean value if a frame was returned at all times. **img** is a frame.
* then, we convert **img** to grayscale


```python
while True:
    # Read the frame
    ret, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display
    cv2.imshow('img', img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
# Release the VideoCapture object
cap.release()
```

# Results

![png](/images/face_id/Screen Shot 2020-06-28 at 7.40.40 PM.png)
![png](/images/face_id/Screen Shot 2020-06-28 at 7.40.51 PM.png)

### As you can see it works, and it is really simple to build. Imagine if you spend on this project like 10 hours. I bet you can build a very sophisticated algorithm.