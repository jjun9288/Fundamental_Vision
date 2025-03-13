# Implementations and applications for fundamental computer vision technologies 

### 1. Camera calibration
- Camera calibration code with OpenCV.

### 2. Canny edge detector
- Canny edge detector code with NumPy
1. Convolution with Sobel filter
2. Compute the gradient and magnitude of the computed images
3. Non maximum suppression
4. Hysteresis threshold

### 3. Image rotation
- Implementation of image rotation (forward, backward, bilinear interpolation)

### 4. Homography
- Stitching two images captured from different views by computing the homography matrix
1. SIFT descriptors to extract feature points that match between two images
2. Bruteforce matcher with K nearest neighbor to get the strong related feature points
3. RANSAC algorithm implementation
4. Compute the homography matrix
5. Image stitching by warping the image