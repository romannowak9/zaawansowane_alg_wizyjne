import cv2
import numpy as np


def compute_points(images_prefix: str):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
    # inner size of chessboard
    width = 9
    height = 6
    square_size = 0.025  # 0.025 meters

    # prepare object points , like (0 ,0 ,0) , (1 ,0 ,0) , (2 ,0 ,0) .... ,(8 ,6 ,0)
    objp = np.zeros((height * width, 1, 3), np.float64)
    objp[:, 0, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * square_size  # Create real world coords . Use your metric .
    # Arrays to store object points and image points from all the images .
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane .
    img_width = 640
    img_height = 480
    image_size = (img_width, img_height)
    path = ""
    image_dir = path + "pairs/"
    number_of_images = 50
    for i in range(1, number_of_images):
        # read image
        img = cv2.imread(image_dir + f"{images_prefix}_%02d.png" % i)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            (width, height),
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        Y, X, channels = img.shape
        # skip images where the corners of the chessboard are too close to the edges of the image
        if ret == True:
            minRx = corners[:, :, 0].min()
            maxRx = corners[:, :, 0].max()
            minRy = corners[:, :, 1].min()
            maxRy = corners[:, :, 1].max()
            border_threshold_x = X / 12
            border_threshold_y = Y / 12
            x_thresh_bad = False
            y_thresh_bad = False
            if minRx < border_threshold_x:
                x_thresh_bad = True
                y_thresh_bad = False
            if minRy < border_threshold_y:
                y_thresh_bad = True
            if (y_thresh_bad == True) or (x_thresh_bad == True):
                continue

        # If found , add object points , image points ( after refining them )
        if ret == True:
            objpoints.append(objp)
            # improving the location of points (sub - pixel )
            corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            # Show the image to see if pattern is found ! imshow function .
            cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            cv2.imshow("Corners", img)
            cv2.waitKey(5)
        else:
            print("Chessboard couldn't detected. Image pair: ", i)
            continue
        
    cv2.destroyAllWindows()
    
    return objpoints, imgpoints, image_size, calibration_flags
    
    
def calibrate_camera(objpoints, imgpoints, image_size, calibration_flags):
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    ret, K, D, _, _ = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        image_size,
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
    # Let ’s rectify our results
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, image_size, cv2.CV_16SC2
    )
    
    return tvecs, rvecs, K, D, map1, map2


def stereo_calibration():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
    # inner size of chessboard
    width = 9
    height = 6
    square_size = 0.025  # 0.025 meters

    # prepare object points , like (0 ,0 ,0) , (1 ,0 ,0) , (2 ,0 ,0) .... ,(8 ,6 ,0)
    objp = np.zeros((height * width, 1, 3), np.float64)
    objp[:, 0, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * square_size  # Create real world coords . Use your metric .
    # Arrays to store object points and image points from all the images .
    objpoints = []  # 3d point in real world space
    l_imgpoints = []  # 2d points in image plane .
    r_imgpoints = []  # 2d points in image plane .
    img_width = 640
    img_height = 480
    image_size = (img_width, img_height)
    path = ""
    image_dir = path + "pairs/"
    number_of_images = 50
    for i in range(1, number_of_images):
        # read image
        l_img = cv2.imread(image_dir + f"left_%02d.png" % i)
        r_img = cv2.imread(image_dir + f"right_%02d.png" % i)
        if l_img is None or r_img is None:
            continue
        
        l_gray = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
        r_gray = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        l_ret, l_corners = cv2.findChessboardCorners(
            l_gray,
            (width, height),
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        r_ret, r_corners = cv2.findChessboardCorners(
            r_gray,
            (width, height),
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        Y, X, channels = l_img.shape
        # skip images where the corners of the chessboard are too close to the edges of the image
        if l_ret == True:
            minRx = l_corners[:, :, 0].min()
            maxRx = l_corners[:, :, 0].max()
            minRy = l_corners[:, :, 1].min()
            maxRy = l_corners[:, :, 1].max()
            border_threshold_x = X / 12
            border_threshold_y = Y / 12
            x_thresh_bad = False
            y_thresh_bad = False
            if minRx < border_threshold_x:
                x_thresh_bad = True
                y_thresh_bad = False
            if minRy < border_threshold_y:
                y_thresh_bad = True
            if (y_thresh_bad == True) or (x_thresh_bad == True):
                continue
            
        if r_ret == True:
            minRx = r_corners[:, :, 0].min()
            maxRx = r_corners[:, :, 0].max()
            minRy = r_corners[:, :, 1].min()
            maxRy = r_corners[:, :, 1].max()
            border_threshold_x = X / 12
            border_threshold_y = Y / 12
            x_thresh_bad = False
            y_thresh_bad = False
            if minRx < border_threshold_x:
                x_thresh_bad = True
                y_thresh_bad = False
            if minRy < border_threshold_y:
                y_thresh_bad = True
            if (y_thresh_bad == True) or (x_thresh_bad == True):
                continue

        # If found , add object points , image points ( after refining them )
        if l_ret and r_ret == True:
            objpoints.append(objp)
            # improving the location of points (sub - pixel )
            l_corners2 = cv2.cornerSubPix(l_gray, l_corners, (3, 3), (-1, -1), criteria)
            l_imgpoints.append(l_corners2)
            r_corners2 = cv2.cornerSubPix(r_gray, r_corners, (3, 3), (-1, -1), criteria)
            r_imgpoints.append(r_corners2)
            # Draw and display the corners
            # Show the image to see if pattern is found ! imshow function .
            # cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            # cv2.imshow("Corners", img)
            # cv2.waitKey(5)
            
        else:
            print("Chessboard couldn't detected. Image pair: ", i)
            continue
        
    cv2.destroyAllWindows()
    
    N_OK = len(objpoints)
    l_K = np.zeros((3, 3))
    l_D = np.zeros((4, 1))
    l_rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    l_tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    l_ret, l_K, r_D, _, _ = cv2.fisheye.calibrate(
        objpoints,
        l_imgpoints,
        image_size,
        l_K,
        l_D,
        l_rvecs,
        l_tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
    # Let ’s rectify our results
    # l_map1, l_map2 = cv2.fisheye.initUndistortRectifyMap(
    #     l_K, l_D, np.eye(3), l_K, image_size, cv2.CV_16SC2
    # )
    
    r_K = np.zeros((3, 3))
    r_D = np.zeros((4, 1))
    r_rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    r_tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    r_ret, r_K, r_D, _, _ = cv2.fisheye.calibrate(
        objpoints,
        r_imgpoints,
        image_size,
        r_K,
        r_D,
        r_rvecs,
        r_tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
    # Let ’s rectify our results
    # r_map1, r_map2 = cv2.fisheye.initUndistortRectifyMap(
    #     r_K, r_D, np.eye(3), r_K, image_size, cv2.CV_16SC2
    # )
    
    return l_K, r_K, l_D, r_D, objpoints, l_imgpoints, r_imgpoints, image_size

