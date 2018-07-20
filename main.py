import numpy as np
import cv2

MIN_MATCHES = 15
HEIGHT = 480
WIDTH = 640

def main():
    
    # Compute model first
    model = cv2.imread('data/model.jpg', 0)
    # Draw a rectangle that marks the found model in the frame
    h, w = model.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    
    # Converting Image
    target = cv2.imread('data/transfer.jpg')
    t_h, t_w, _ = target.shape
    target_pts = np.float32([[0,0], [0, t_h-1], [t_w-1, t_h-1], [t_w-1, 0]]).reshape(-1, 1, 2)

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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

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
                # assuming matches stores the matches found and 
                # returned by bf.match(des_model, des_frame)
                # differenciate between source points and destination points
                src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                # compute Homography
                _M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                # project corners into frame
                dst = cv2.perspectiveTransform(pts, _M)

                # target image
                M = cv2.getAffineTransform(target_pts[:3], dst[:3])

                # img = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

                _target = cv2.warpAffine(target, M, (WIDTH, HEIGHT))
                img = cv2.addWeighted(_target, 0.5, frame, 0.5, 0)
                cv2.imshow('frame', img)
                cv2.imshow('target', _target)
            else:
                print("Not enough matches have been found - {} / {}".format( len(matches), MIN_MATCHES))
                # show result
                cv2.imshow('frame', frame)
        else:
            print("taget has no features!")
            cv2.imshow('frame', frame)
        
        key = cv2.waitKey(100)

        if key == ord('q'):  # exit on `q`
            cap.release()
            break
        if key == ord('p'):  # print image
            if img.any():
                cv2.imwrite('frame_with_target.jpg', img)
            if _target.any():
                cv2.imwrite('target_wrapped.jpg', _target)
            if frame.any():
                cv2.imwrite('frame.jpg', frame)
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()