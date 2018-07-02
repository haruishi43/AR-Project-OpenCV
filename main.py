import numpy as np
import cv2

MIN_MATCHES = 15
HEIGHT = 480
WIDTH = 640

def cameraPoseFromHomography(H):
    H1 = H[:, 0]
    H2 = H[:, 1]
    H3 = np.cross(H1, H2)

    norm1 = np.linalg.norm(H1)
    norm2 = np.linalg.norm(H2)
    tnorm = (norm1 + norm2) / 2.0;

    T = H[:, 2] / tnorm
    return np.mat([H1, H2, H3, T])


def project(p, mat):
    x = mat[0][0] * p[0] + mat[0][1] * p[1] + mat[0][2] * p[2] + mat[0][3] * 1
    y = mat[1][0] * p[0] + mat[1][1] * p[1] + mat[1][2] * p[2] + mat[1][3] * 1
    w = mat[3][0] * p[0] + mat[3][1] * p[1] + mat[3][2] * p[2] + mat[3][3] * 1
    return np.array([WIDTH * (x / w + 1) / 2., HEIGHT - HEIGHT * (y / w + 1) / 2.])


def norm2(a, b):
    a = a[0]
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return dx * dx + dy * dy


def evaluate(mat, pts, ref): 
    c0 = project(ref[0], mat)
    c1 = project(ref[1], mat)
    c2 = project(ref[2], mat)
    c3 = project(ref[3], mat)
    return norm2(pts[0], c0) + norm2(pts[1], c1) + norm2(pts[2], c2) + norm2(pts[3], c3)


def perturb(mat, amount):
    from copy import deepcopy
    from random import randrange, uniform
    mat2 = deepcopy(mat)
    mat2[randrange(4)][randrange(4)] += uniform(-amount, amount)
    return mat2


def approximate(mat, pts, ref, amount, n=100000):
    est = evaluate(mat, pts, ref)

    for i in range(n):
        mat2 = perturb(mat, amount)
        est2 = evaluate(mat2, pts, ref)
        if est2 < est:
            mat = mat2
            est = est2

    return mat, est


def estimateProjectionnMatrix(pts):
    mat = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    ref = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 1],
    ])

    for i in range(10):
        print(i)
        mat, _ = approximate(mat, pts, ref, 1)
        mat, _ = approximate(mat, pts, ref, .1)

    print(mat)

    return mat


def main():
    
    # Compute model first
    model = cv2.imread('data/model.jpg', 0)
    # Draw a rectangle that marks the found model in the frame
    h, w = model.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    
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

    vis = False

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
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                # project corners into frame
                dst = cv2.perspectiveTransform(pts, M)

                # P = cameraPoseFromHomography(M)  # output = [R1, R2, R3, T]
                if vis:
                    # connect them with lines
                    img = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA) 
                    cv2.imshow('frame', img)
                else:
                    cv2.imshow('frame', frame)
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
        elif key == ord('e'):  # estimate
            cap.release()
            cv2.destroyWindow('frame')
            break
            vis = True

    
    cv2.destroyAllWindows()
    mat = estimateProjectionnMatrix(dst)
    
    
    
        

if __name__ == "__main__":
    main()
