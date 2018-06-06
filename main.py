import numpy as np
import cv2

MIN_MATCHES = 20


def main():
    
    # Compute model first
    model = cv2.imread('data/model.jpg', 0)

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
                # draw first 15 matches.
                gray_frame = cv2.drawMatches(model, kp_model, gray_frame, kp_frame, matches[:MIN_MATCHES], 0, flags=2)
                
                # assuming matches stores the matches found and 
                # returned by bf.match(des_model, des_frame)
                # differenciate between source points and destination points
                src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                # compute Homography
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                # Draw a rectangle that marks the found model in the frame
                h, w = model.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                # project corners into frame
                dst = cv2.perspectiveTransform(pts, M)  
                # connect them with lines
                img2 = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA) 
                cv2.imshow('frame', img2)
            else:
                print("Not enough matches have been found - {} / {}".format( len(matches), MIN_MATCHES))
                # show result
                cv2.imshow('frame', frame)
        else:
            print("taget has no features!")
            cv2.imshow('frame', frame)
        
        if cv2.waitKey(150) == ord('q'): # exit on `q`
            break
        

if __name__ == "__main__":
    main()