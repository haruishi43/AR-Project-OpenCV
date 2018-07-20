import numpy as np
import cv2

MIN_MATCHES = 20

def main():
    
    # Compute model first
    model = cv2.imread('../data/model.jpg', 0)

    # ORB keypoint detector
    orb = cv2.ORB_create()              
    # create brute force  matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  
    # Compute model keypoints and its descriptors
    kp_model, des_model = orb.detectAndCompute(model, None)  

    # safe guard
    if des_model is None:
        print("no model features!")

    # run camera:
    cap = cv2.VideoCapture(0)
    # set to vga format
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # caputre loop
    while(True):
        
        ret, frame = cap.read()

        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Compute scene keypoints and its descriptors
        kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)
        if des_frame is not None:
            # Match frame descriptors with model descriptors
            matches = bf.match(des_model, des_frame)
            # Sort them in the order of their distance
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) > MIN_MATCHES:
                gray_frame = cv2.drawKeypoints(gray_frame, kp_frame, None, color=(0,255,0), flags=0)
                model_frame = cv2.drawKeypoints(model, kp_model, None, color=(0,255,0), flags=0)
                cv2.imshow('image', model_frame)
                cv2.imshow('frame', gray_frame)
            else:
                print("Not enough matches have been found - {} / {}".format( len(matches), MIN_MATCHES))
                # show result
                cv2.imshow('frame', gray_frame)
        else:
            print("taget has no features!")
            cv2.imshow('frame', gray_frame)
        
        key = cv2.waitKey(100)

        if key == ord('q'): # exit on `q`
            break
        if key == ord('p'):  # print image
            if gray_frame.any():
                cv2.imwrite('features_frame.jpg', gray_frame)
            if model_frame.any():
                cv2.imwrite('features_model.jpg', model_frame)
        

if __name__ == "__main__":
    main()