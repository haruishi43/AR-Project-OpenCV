# Project as a whole:

What I'll be crating is a demo application for rendering 3D object onto a surface (such as cards or ar-markers).

Objectives:

- Render onto the screen a video frame with a 3D object of a figure which the position and orientation matches of that of a predefined flat surface
  - predefined flat surface should have enough 'features' (using business cards or ar-marker should be sufficient)
  - The position and orientation should change in real-time

## Steps:

1. Identify flat surface in the image frame (from rendered video) that is the same as the reference image
2. Estimate homography using the transformation from the reference image frame to the target image
3. Derive coordinate transformation (projection) from reference to target coordinate system
4. Render the 3D object in the target image

## Working with the target surface:

First of all, we need to recognize the target surface.
There are many techniques that could be used to solve this problem, but feature based recognition is simple and the method that I had learned from the Computer Vision class.
The steps for this method is:
- feature detection (detector)
- feature description (descriptor)
- feature matching

### Feature Detection:

In computer vision, the conecpt of feature detection referes to methods that aim to compute abstractions of image information which are given by points or groups of points in images. 
Feature detection in general means finding interesting points (features) in the image such as corners, templates, edges, and so on.
So by looking in both the reference and target images for features that are similar and stands out, we can later match those features to find the reference image inside of the target image (i.e., card, AR marker).
For this project, it is assumed that the same object is found when there are enough features that are matched.

Conditions:

- Reference image should show only the object that you want to track
- Dimensions of the reference image should be given
- Feature in the images should be unique (such as corners or edges), and the object in the reference should be invariant

### Feature Description:

In pattern recognition, feature extraction is a special form of dimensionality reduction. 
When the input image to an algorithm is too large to be processed and it is suspected to be notoriously redundant, then the input image will be transformed into a reduced representation set of features (feature vector).
This process of turning the input image into a set of features is called feature extraction.
In short, feature extraction represets the interesting points found by feature detector and compare them with other feature points in the image to reduce the representation.

There are many algorithms that extract image features and compute its descriptors such as SIFT, SURF, or Harris.
The one that we will use in this project is OpenCV's native algorithm, the ORB (Oriented FAST and Rotated BRIEF).
This will produce binary strings as its descriptor.

```python

img = cv2.imread('some_image.png', 0)

orb = cv2.ORB_create()  # initialize detector
kp = orb.detect(img, None)  # find keypoints
kp, des = orb.compute(img, kp)  # compute descriptors

img2 = cv2.drawKeypoints(img, kp, img, color=(0, 255, 0), flags=0)
cv2.imshow('keypoints', img2)
cv2.waitKey(0)
```

### Feature Matching:












