import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Image preprocessing
root = os.getcwd()
img_path = os.path.join(root, 'data/')
img_1 = cv2.imread(img_path + 'school_1.jpg')
img_2 = cv2.imread(img_path + 'school_2.jpg')

img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

# Finding keypoints and descriptors

'''
orb = cv2.ORB_create()
keypoints_1, descriptor_1 = orb.detectAndCompute(img_1_gray, None)
keypoints_2, descriptor_2 = orb.detectAndCompute(img_2_gray, None)
'''

sift = cv2.SIFT_create()
keypoints_1, descriptor_1 = sift.detectAndCompute(img_1_gray, None)
keypoints_2, descriptor_2 = sift.detectAndCompute(img_2_gray, None)

# Visualizing keypoints
key_img_1 = cv2.drawKeypoints(img_1_gray, keypoints_1, np.array([]), (0, 0, 255))
key_img_2 = cv2.drawKeypoints(img_2_gray, keypoints_2, np.array([]), (0, 0, 255))

cv2.imwrite('keypoint_1_sift.jpg', key_img_1)
cv2.imwrite('keypoint_2_sift.jpg', key_img_2)

# Matching descriptors using Bruteforce matching with Hamming distance

'''
bruteforce = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bruteforce.match(descriptor_1, descriptor_2, k=2)
'''

bruteforce = cv2.BFMatcher()
matches = bruteforce.knnMatch(descriptor_1, descriptor_2, k=2)

good_match = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_match.append(m)

src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in good_match]).reshape(-1,1,2)
dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in good_match]).reshape(-1,1,2)

# RANSAC algorithm implementation to compute the Homography matrix
NUM_ITERATION = 100000
THRESHOLD = 5

best_inlier_cnt = 0
best_homography = 0

for _ in range(NUM_ITERATION):
    # Randomly select 4 matching points
    random_idx = np.random.choice(len(good_match), 4, replace=False)
    src_pts_rand = src_pts[random_idx,:]
    dst_pts_rand = dst_pts[random_idx,:]

    # Computing homography with these points
    A = []
    for i in range(4):
        x, y = src_pts_rand[i][0]
        u, v = dst_pts_rand[i][0]
        A.append([x, y, 1, 0, 0, 0, -x*u, -y*u, -u])
        A.append([0, 0, 0, x, y, 1, -x*v, -y*v, -v])
    A = np.array(A)

    _, _, V = np.linalg.svd(A)
    h = V[-1, :]
    H = h.reshape((3,3))

    # Transform the source points using the computed Homography matrix
    ones = np.ones((src_pts.shape[0], 1))
    src_pts_1 = np.concatenate((src_pts, ones[...,np.newaxis]), axis=2)
    dst_pts_1 = np.concatenate((dst_pts, ones[...,np.newaxis]), axis=2)
    src_pts_transformed = np.matmul(H, src_pts_1.swapaxes(1,2)).swapaxes(1,2)
    src_pts_transformed /= src_pts_transformed[:,:,2][:, np.newaxis]

    # Computing the Euclidean distance between the transformed source points and the destination points
    dist = np.abs(np.sum(dst_pts_1 - src_pts_transformed))

    inliers = np.where(dist < THRESHOLD)
    inlier_cnt = len(inliers[0])

    if inlier_cnt > best_inlier_cnt:
        best_inlier_cnt = inlier_cnt
        best_homography = H/H[-1,-1]
        

h1, w1 = img_1.shape[:2]
h2, w2 = img_2.shape[:2]
pts_1 = np.float32([[0, 0], [0, h1], [w1,h1], [w1,0]]).reshape(-1,1,2)
pts_2 = np.float32([[0, 0], [0, h2], [w2,h2], [w2,0]]).reshape(-1,1,2)
pts_2 = cv2.perspectiveTransform(pts_2, best_homography)
pts = np.concatenate((pts_1, pts_2), axis=0)

[xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
[xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
t = [-xmin, -ymin]

Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

result = cv2.warpPerspective(img_1, Ht.dot(best_homography), (xmax-xmin, ymax-ymin))
result[t[1]:h1+t[1], t[0]:w1+t[0]] = img_2

cv2.imwrite('panorama.jpg', result)