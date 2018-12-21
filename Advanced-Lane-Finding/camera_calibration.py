import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from util import *
global __DEBUG__

## 3 - D
objpoints = []

## 2 -D
imgpoints = []

##
mtx = None
dist = None


def calibrate_camera():
    images = glob.glob("./camera_cal/*.jpg")
    #img =  mpimg.imread("./camera_cal/calibration1.jpg")
    #plt.imshow(img)
    #plt.show()
    ### note 8 6 will change

    x_points_num = 9
    y_points_num = 6
    for img_path in images:
        if img_path == "./camera_cal/calibration5.jpg":
            continue
        img = mpimg.imread(img_path)

        objp = np.zeros((x_points_num * y_points_num, 3), np.float32)
        objp[:,:2] = np.mgrid[0:x_points_num, 0:y_points_num].T.reshape(-1, 2)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (x_points_num, y_points_num), None)

        if ret == True:
            print(img_path + " alright")
            imgpoints.append(corners)
            objpoints.append(objp)
            img = cv2.drawChessboardCorners(img, (x_points_num,y_points_num), corners, ret)

            #plt.imshow(img)
            #plt.show()
        else:
            print(img_path + " not right")
            continue

    global dist, mtx
    ret, mtx, dist, rvecs, tvecs \
            = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        #plt.imshow(dst)
        #plt.show()


# Define a function that takes an image, number of x and y points,
# camera matrix and distortion coefficients
def corners_unwarp(img, nx, ny, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        # If we found corners, draw them! (just for fun)
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100 # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
                                     [img_size[0]-offset, img_size[1]-offset],
                                     [offset, img_size[1]-offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)
        #show_img_2(warped)
    # Return the resulting image and matrix
    return warped, M


def color_and_gradient_pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary

def get_warped_img(img_path):
    img =  mpimg.imread(img_path)

    undistort_img = cv2.undistort(img, mtx, dist, None, mtx)

    #color_and_gradient_pipeline(undistort_img)
    #undistort_img = color_and_gradient_filter(undistort_img)

    img_size = (img.shape[1], img.shape[0])

    src_pts = [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
         [((img_size[0] / 6) - 10), img_size[1]],
         [(img_size[0] * 5 / 6) + 60, img_size[1]],
         [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]]

    src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
         [((img_size[0] / 6) - 10), img_size[1]],
         [(img_size[0] * 5 / 6) + 60, img_size[1]],
         [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])

    for src_pt in src_pts:
        x = src_pt[0]
        y = src_pt[1]
        print(x)
        print(y)
        cv2.circle(undistort_img, (int(x), int(y)), 10, [255, 0, 0], thickness=10, lineType=8,
                   shift=0)

    show_img(undistort_img)

    dst = np.float32(
        [[(img_size[0] / 4), 0],
         [(img_size[0] / 4), img_size[1]],
         [(img_size[0] * 3 / 4), img_size[1]],
         [(img_size[0] * 3 / 4), 0]])

    ### TODO: make the image in 2 channel

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undistort_img, M, img_size, flags = cv2.INTER_LINEAR)
    #warped = color_and_gradient_filter(warped)
    show_img(warped)



## TODO
if 1 == 0:
    calibrate_camera()
    img_file = './camera_cal/calibration16.jpg'
    test_img = mpimg.imread(img_file)
    corners_unwarp(test_img, 9, 6, mtx, dist)
    ##get_warped_img("/Users/xingwang/udacity/udacity_self_driving_car_p1/home/CarND-LaneLines-P1/test_images/solidWhiteRight.jpg")
    get_warped_img("/Users/xingwang/test1.jpg")