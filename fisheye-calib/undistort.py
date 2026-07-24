import cv2
import numpy as np
import os
import glob

K = np.array([[500.26760819, 0, 616.81375815], [0, 500.13207688, 373.32708663], [0, 0, 1]])
D = np.array([[-0.02142327], [-0.00168711], [0.00210896], [-0.00131498]])

def undistort(img, balance, dim2):
    dim1 = img.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort (must match calibration image size)
    assert dim1[0]/dim1[1] == dim2[0]/dim2[1], "Image to undistort must have same aspect ratio as the calibration images"
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim1, np.eye(3), balance=balance, new_size=dim2)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim2, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


image_files = glob.glob('calib_images/*.jpg')

for fn_name in image_files:
    img = cv2.imread(fn_name)
    undistorted_img = undistort(img, balance=0.8, dim2=(1920, 1080))
    if not os.path.exists('undistorted_images'):
        os.makedirs('undistorted_images')
    cv2.imwrite('undistorted_images/' + os.path.basename(fn_name), undistorted_img)
    cv2.waitKey(0)
