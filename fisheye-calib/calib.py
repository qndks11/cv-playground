import cv2
import numpy as np
import os
import glob

pattern_size = (6, 8)  # Number of inner corners per a chessboard row and column (row, column)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
calibration_flags = cv2.CALIB_RECOMPUTE_EXTRINSIC + cv2.CALIB_CHECK_COND + cv2.CALIB_FIX_SKEW

obj_points = []
img_points = []

objp = np.zeros((1, pattern_size[0]*pattern_size[1], 3), np.float32)           # (1, N, 3)
objp[0,:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) # grid coordinates (0,0,0), (1,0,0), (2,0,0), ..., (5,7,0)

image_files = glob.glob('calib_images/*.jpg')
for fn_name in image_files:
    image_file = cv2.imread(fn_name)
    gray = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
    retval, corners = cv2.findChessboardCorners(gray, pattern_size, None) # findChessboardCorners returns corner points
    if retval:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) # refine corner locations to subpixel accuracy
        obj_points.append(objp)                        # (1, N, 3)
        img_points.append(corners2.reshape(1, -1, 2))  # (1, N, 2)
        corner_img = cv2.drawChessboardCorners(image_file, pattern_size, corners2, retval) # draw the corners on the image
        if not os.path.exists('chessboard_corners'):
            os.makedirs('chessboard_corners')
        cv2.imwrite(f'chessboard_corners/{os.path.basename(fn_name)}', corner_img)
        cv2.waitKey(0)
    else:
        print(f"Chessboard corners not found in image: {fn_name}")
        
ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    obj_points,
    img_points,
    gray.shape[::-1],
    None,          # K
    None,          # D
    flags=calibration_flags,
    criteria=criteria
)

print("Camera matrix (K):")
print(K)
print("Distortion coefficients (D):")
print(D)        
print("Rotation vectors (rvecs):")
print(rvecs)
print("Translation vectors (tvecs):")
print(tvecs)