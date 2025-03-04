import numpy as np
import cv2
import glob

# Defining the dimension of checkerboard
CHECKERBOARD = (9,6) #체커 보드 행/열 당 내부 코너 수
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Defining world coordinates
#size_of_chessboard_squares_mm = 20
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)   # 3차원 (CHECKERBOARD[0]*CHECKERBOARD[1])x3 matrix 1개 생성. 즉 각 CHECKERBOARD point마다 (x,y,z) 생성하기 위해 만드는 것.
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)  # 각 꼭지점마다 죄표 생성 (0,0,0) ~ (5,8,0)
#objp = objp * size_of_chessboard_squares_mm
previous_img_shape = None

# World 좌표계의 3D 점들을 저장할 리스트
object_point = []

# 실제 2D 이미지와 코너 좌표들을 저장할 리스트
image_point = []

images = glob.glob('*.jpg')

for i in images:
    img = cv2.imread(i)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Checkerboard 코너 찾기 (원하는 갯수의 코너가 발견되면 ret = True)
    ret, corners = cv2.findChessboardCorners(gray_img, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret == True:
        # 각각 3D 좌표, 2D 좌표를 append
        object_point.append(objp)
        corners2 = cv2.cornerSubPix(gray_img, corners, (11,11), (-1,-1), criteria)    #cornerSubPix는 원본 이미지와 코너 위치를 가져와서 원래 위치의 작은 이웃 내에서 가장 좋은 코너 위치를 찾는 함수
        image_point.append(corners2)
        
        # 코너 그리기 및 표시
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(0)
    
cv2.destroyAllWindows()

##################################################################################################################
# 이제 3D, 2D 포인트를 알고있으니 이를 내부 파라미터로 넣어 왜곡된 촬영 영상을 보정할 수 있게 되는 것이다.
##################################################################################################################


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_point, image_point, (300,168), None, None)

print('Intrinsic 행렬 : ', mtx)
print('렌즈 왜곡 계수 : ', dist)
print('회전 벡터 : ', rvecs)
print('이동 벡터 : ', tvecs)



img = cv2.imread('./newimage.jpg')
h, w = img.shape[:2]

# Camera matrix를 수정해주는 함수
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# 왜곡을 제거해주는 함수
dst = cv2.undistort(img, mtx, dist, None, new_camera_matrix)

x,y,w,h = roi

dst = dst[y:y+h, x:x+w]

# 데이터셋에 있는 이미지들을 통해 패턴을 분석하고, 분석한 패턴을 통해 camera matrix를 얻게 된다. Matrix를 개선한 뒤 최종적으로 왜곡된 부분을 제거해서
# result로 저장한다.

cv2.imwrite('calib.png', dst)

# Camera matrix와 왜곡 계수를 저장하여 추후에 계속 사용 가능
np.savez('calib.npz', ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

mean_error = 0

for i in range(len(object_point)):
    image_point2,_ = cv2.projectPoints(object_point[i], rvecs[i], tvecs[1], mtx, dist)
    
    error = cv2.norm(image_point[i], image_point2, cv2.NORM_L2)/len(image_point2)
    mean_error += error

# 평균 reprojection error    
print('Total error : {0}'.format(mean_error/len(object_point)))