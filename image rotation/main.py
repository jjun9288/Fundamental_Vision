from pickletools import uint8
from turtle import width
import cv2
import numpy as np
import math

path = 'C:/Users/jjun8/Desktop/VCLab/수업/1-1/컴퓨터비전/rotation/'
img = cv2.imread('rotation/lena.jpg')

h, w, c = img.shape

# rotation matrix
angle = -15

cos = np.cos(-angle * (math.pi/180))
sin = np.sin(-angle * (math.pi/180))

r = np.array([[cos, -sin],
              [sin, cos]])

r_inv = np.array([[cos, sin],
                  [-sin, cos]])

#--------------------------Forward---------------------------------------------------------------
new_h, new_w = round(abs(h*cos) + abs(w*sin)), round(abs(w*cos) + abs(h*sin))

original_center_height = round(((h+1)/2)-1)
original_center_width = round(((w+1)/2)-1)

new_center_h = round(((new_h+1)/2)-1)
new_center_w = round(((new_w+1)/2)-1)

black = np.zeros(shape=(new_h, new_w, 3))

for i in range(h): 
    for j in range(w):
        y = i-original_center_height
        x = j-original_center_width
        
        new_y = round(y*cos - x*sin)
        new_x = round(y*sin + x*cos)
        
        new_y += new_center_h
        new_x += new_center_w
        
        if 0 <= new_x < new_w and 0 <= new_y < new_h:
            black[new_y, new_x, :] = img[i, j, :]

            
black = np.array(black, dtype = np.uint8)
cv2.imshow('forward', black)
#cv2.imshow('lena', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#--------------------------Backward--------------------------------------------------------------

black = np.zeros(shape=(new_h, new_w, 3))

for i in range(new_h):
    for j in range(new_w):
        y = i - new_center_h
        x = j - new_center_w
        
        original_y = round(y*cos + x*sin)
        original_x = round(-y*sin + x*cos)
        
        original_y += original_center_height
        original_x += original_center_width
        
        
        if 0 <= original_x < w and 0 <= original_y < h:
            black[i, j, :] = img[original_y, original_x, :]
        
black = np.array(black, dtype = np.uint8)
cv2.imshow('backward', black)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#--------------------------Interpolation---------------------------------------------------------       
black = np.zeros(shape=(new_h, new_w, 3))
 
for i in range(new_h):
    for j in range(new_w):
        y = i - new_center_h    
        x = j - new_center_w    
        
        original_y = int(y*cos + x*sin)
        original_x = int(-y*sin + x*cos)
        
        original_y += original_center_height
        original_x += original_center_width
        

        if 0 <= original_x < w-1 and 0 <= original_y < h-1:
            
            real_y = y*cos + x*sin
            real_x = -y*sin + x*cos
            
            real_y += original_center_height
            real_x += original_center_width
            
            part_y = real_y - int(real_y)
            part_x = real_x - int(real_x)
            #print(part_y, part_x)
            black[i,j] = part_x*((part_y*img[original_y+1, original_x+1]) + ((1-part_y)*img[original_y, original_x+1])) + (1-part_x)*((part_y*img[original_y+1, original_x]) + ((1-part_y)*img[original_y, original_x]))
                
black = np.array(black, dtype = np.uint8)
cv2.imshow('interpolation', black)
cv2.waitKey(0)
cv2.destroyAllWindows()

