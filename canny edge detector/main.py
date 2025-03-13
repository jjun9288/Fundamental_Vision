import os
import cv2
import numpy as np
import math

'''
영상 입력 부분
'''
#------------------Reading Video-------------------
path = 'C:/Users/jjun8/Desktop/test/'
cap = cv2.VideoCapture(path + 'data/swan.mp4')
ret, frame = cap.read()
h,w,l = frame.shape
#--------------------------------------------------


#--------Sobel mask filter-------------
sobel_gx = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]])

sobel_gy = np.array([[-1,-2,-1],
                    [0,0,0],
                    [1,2,1]])
#--------------------------------------


def rgb2gray(img):
    # RGB 3채널 이미지를 Grayscale 이미지로 변형
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray_img = (0.2989*r + 0.5870*g + 0.1140*b)
    return gray_img

def padding(image, filter):
    # convolution 연산 후에도 연산 전 이미지의 shape과 맞추기 위해 진행
    '''
    image : grayscale 이미지
    filter : derivative filter
    '''
    image_h, image_w = image.shape
    pad_h, pad_w = (filter.shape[0]-1)//2, (filter.shape[1]-1)//2
    result = np.zeros(shape=(image_h+(2*pad_h), image_w+(2*pad_w)))
    for i in range(pad_h,image_h+pad_h):
        for j in range(pad_w,image_w+pad_w):
            result[i][j] = image[i-pad_h][j-pad_w]
    return result

def convolution(img, pad_img, filter): 
    # convolution 연산 수행하는 함수
    '''
    img : 원본 이미지
    pad_img : zero-padding을 한 grayscale 이미지
    filter : derivative filter
    '''
    img_h, img_w, l = img.shape
    filter_h, filter_w = filter.shape
    result = np.zeros(shape=(img_h, img_w))
    for i in range(img_h):
        for j in range(img_w):
            sum = 0
            for k in range(filter_h):
                for l in range(filter_w):
                    sum += pad_img[k+i][l+j] * filter[k][l]
            result[i][j] = sum
    return result

def magnitude_grad(img_1, img_2):
    #sobel x,y 필터 연산 수행한 두 이미지의 magnitude, gradient 구하는 함수
    '''
    img_1 : sobel_gx랑 convolution한 이미지
    img_2 : sobel_gy랑 convolution한 이미지
    '''
    h, w = img_1.shape
    result = np.zeros(shape=(h,w))
    theta = np.zeros(shape=(h,w))
    for i in range(h):
        for j in range(w):
            result[i][j] = ((img_1[i][j]**2 + img_2[i][j]**2)**(1/2))    
            theta[i][j] = (math.atan2(img_2[i][j], img_1[i][j]))*180/math.pi
            if theta[i][j] < 0:
                theta[i][j] = -theta[i][j]
    return result, theta

def grad_nms(img, theta):
    #gradient를 이용한 non max suppression을 수행하는 함수. 기울기를 이용하여 해당 기울기 방향으로 이웃한 두 픽셀과 비교
    h, w = img.shape
    dx = [1,0,-1,0,1,1,-1,-1]
    dy = [0,1,0,-1,1,-1,-1,1]
    result = np.zeros(shape=(h,w))
    for i in range(1,h-1):
        for j in range(1,w-1):
            if (0 <= theta[i][j]  < 22.5) or (157.5 < theta[i][j] <= 180):
                comp_1, comp_2 = img[i][j+1], img[i][j-1]
            elif (22.5 <= theta[i][j] < 67.5):
                comp_1, comp_2 = img[i+1][j-1], img[i-1][j+1]
            elif (67.5 <= theta[i][j] < 112.5):
                comp_1, comp_2 = img[i+1][j], img[i-1][j]
            elif (112.5 <= theta[i][j] <= 157.5):
                comp_1, comp_2 = img[i-1][j-1], img[i+1][j+1]
            
            if (img[i][j] > comp_1 and img[i][j] > comp_2):
                result[i][j] = img[i][j]
            else:
                result[i][j] = 0  
    return result

def nms(img):
    #두번째 방식의 non max suppression. 인접 픽셀값 중 본인보다 큰 값이 있다면 0으로 만들어 edge를 더 얇게 만들어주는 함수
    h, w = img.shape
    dx = [1,0,-1,0,1,1,-1,-1]
    dy = [0,1,0,-1,1,-1,-1,1]
    for i in range(h):
        for j in range(w):
            tmp = img[i][j]
            for k in range(len(dx)):
                nx = i + dx[k]
                ny = j + dy[k]
                if 0<=nx<h and 0<=ny<w:
                    comp = img[nx][ny]
                    if tmp < comp:
                        img[i][j] = 0
    return img

def threshold(img, param_1, param_2):
    # Hysteresis Thresholding. max threshold, min threshold를 지정하고 max threshold보다 큰 값은 두고, min threshold보다 작은 값은
    # 없앤다. 그 사이에 있는 값은 나중에 edge tracking을 위해 놔둔다.
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if img[i][j] > param_1:
                img[i][j] = 255
            elif param_2 < img[i][j] <= param_1:
                img[i][j] = 25
            else:
                img[i][j] = 0
    return img

def hysteresis(img):
    #Threshold를 통해 max threshold와 min threshold 사이에 있는 값에 대하여 edge tracking 수행. 인접한 8개 픽셀과 비교하여
    #강한 값이 인접해있다면 해당 값을 강한 값으로 바꿔주고, 없다면 약한 값으로 바꿔준다.
    h, w = img.shape
    dx = [1,0,-1,0,1,1,-1,-1]
    dy = [0,1,0,-1,1,-1,-1,1]
    for i in range(h):
        for j in range(w):
            if img[i][j] == 25:
                for k in range(len(dx)):
                    nx = i + dx[k]
                    ny = j + dy[k]
                    if 0<=nx<h and 0<=ny<w:
                        if img[nx][ny] == 255:
                            img[i][j] = 255
                            continue
                        else:
                            img[i][j] = 0
    return img

'''
영상 출력 부분
원하는 영상 부분의 주석을 제거하시면 됩니다.
'''
# Result
#sobel_out = cv2.VideoWriter(path+'results/sobel.mp4', cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (w,h),0)
#nms_out = cv2.VideoWriter(path+'results/nms.mp4', cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (w,h),0)
#hys_out = cv2.VideoWriter(path+'results_2/90_10.mp4', cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (w,h),0)


# Saving each processed pixels
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        #--------------------Image preprocessing-------------------------------
        gray_img = rgb2gray(frame)
        pad_img = padding(gray_img, sobel_gx)
        #----------------------------------------------------------------------
        
        '''
        sobel필터 처리한 영상을 원하면  이 부분의 주석을 제거하시면 됩니다. 
        '''
        #---------------------Sobel mask--------------------------------
        #sobel_x = convolution(frame, pad_img, sobel_gx)
        #sobel_y = convolution(frame, pad_img, sobel_gy) 
        #sobel, s_theta = magnitude_grad(sobel_x, sobel_y)
        #result = sobel.astype(np.uint8)
        #sobel_out.write(result)
        #----------------------------------------------------------------------
        
        '''
        Non Maximum Suppression 처리한 영상을 원하면 이 부분의 주석을 제거하시면 됩니다.
        '''
        #---------------------Non Maximum Suppression--------------------------
        #sobel_x = convolution(frame, pad_img, sobel_gx)
        #sobel_y = convolution(frame, pad_img, sobel_gy) 
        #sobel, s_theta = magnitude_grad(sobel_x, sobel_y)
        #nms_img = grad_nms(sobel, s_theta)
        #result = nms_img.astype(np.uint8)
        #nms_out.write(result)
        #----------------------------------------------------------------------
        
        '''
        Hysteresis Threshold 처리한 영상을 원하면 이 부분의 주석을 제거하시면 됩니다.
        '''
        #---------------------Hysteresis Thresholding--------------------------
        #sobel_x = convolution(frame, pad_img, sobel_gx)
        #sobel_y = convolution(frame, pad_img, sobel_gy) 
        #sobel, s_theta = magnitude_grad(sobel_x, sobel_y)
        #nms_img = grad_nms(sobel, s_theta)
        #double = threshold(nms_img, 90, 10)
        #result = hysteresis(double)
        #result = result.astype(np.uint8)
        #hys_out.write(result)
        #----------------------------------------------------------------------
        
    else:
        break
'''
마지막으로 원하는 영상(Sobel, NMS, Hysteresis) 중 원하는 영상의 주석을 제거하고 실행하시면 됩니다.
'''   
#sobel_out.release()
#nms_out.release()
#hys_out.release()
print('done')