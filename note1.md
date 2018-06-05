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




