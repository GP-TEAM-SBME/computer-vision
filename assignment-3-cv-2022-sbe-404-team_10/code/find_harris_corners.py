"""
This is a script to demonstrate the implementation of Harris corner function to
detect corners from the image.

"""

import argparse
import cv2
import numpy as np

def find_harris_corners(input_img, k, window_size, threshold):
    
    corner_list = []
    output_img = cv2.cvtColor(input_img.copy(), cv2.COLOR_GRAY2RGB)
    
    offset = int(window_size/2)
    y_range = input_img.shape[0] - offset
    x_range = input_img.shape[1] - offset
    
    
    dy, dx = np.gradient(input_img)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    
    
    for y in range(offset, y_range):
        for x in range(offset, x_range):
            
            #Values of sliding window
            start_y = y - offset
            end_y = y + offset + 1
            start_x = x - offset
            end_x = x + offset + 1
            
            #The variable names are representative to 
            #the variable of the Harris corner equation
            windowIxx = Ixx[start_y : end_y, start_x : end_x]
            windowIxy = Ixy[start_y : end_y, start_x : end_x]
            windowIyy = Iyy[start_y : end_y, start_x : end_x]
            
            #Sum of squares of intensities of partial derevatives 
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            #Calculate determinant and trace of the matrix
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            
            #Calculate r for Harris Corner equation
            r = det - k*(trace**2)

            if r > threshold:
                corner_list.append([x, y, r])
                output_img[y,x] = (0,0,255)
    
    return corner_list, output_img 

def harris(input_img):
    
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    k = 0.04
    window_size = 5
    threshold = 10000.00
    
    # if args.k:
    #     k = args.k
    
    # if args.window_size:
    #     window_size = args.window_size
        
    # if args.threshold:
    #     threshold = args.threshold

    if input_img is not None:
        
        print ("Detecting Corners Started!")
        corner_list, corner_img = find_harris_corners(input_img, k, window_size, threshold)
        corner_file = open('corners_list.txt', 'w')
        corner_file.write('x ,\t y, \t r \n')
        for i in range(len(corner_list)):
            corner_file.write(str(corner_list[i][0]) + ' , ' + str(corner_list[i][1]) + ' , ' + str(corner_list[i][2]) + '\n')
        corner_file.close()
        
        # if corner_img is not None:
        #     cv2.imwrite("corners_img.png", corner_img)
        print ("Detecting Corners Complete!")
        return corner_img
    else:
        print ("Error in input image!")