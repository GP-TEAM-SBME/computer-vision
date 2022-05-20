import numpy as np
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
import time

import math
import cv2
import os
import cmath
from math import log, ceil


def sub2d(img, value):    
    N, M = img.shape
    result = np.zeros((N,M))
    for row in range(N):
	    for col in range(M):
		    result[row][col] = img[row][col] - value
    return result


def add2d(img, value):    
    N, M = img.shape
    result = np.zeros((N,M))
    for row in range(N):
	    for col in range(M):
		    result[row][col] = img[row][col] + value        
    return result
    
def min1d(arr):     
    minElement = arr[0] 
    for i in range(len(arr)):
        if (arr[i] < minElement):
            minElement = arr[i]
    return minElement   

def max1d(arr):     
    maxElement = arr[0] 
    for i in range(len(arr)):
        if (arr[i] > maxElement):
            maxElement = arr[i]
    return maxElement 
    
def flatten(img):
    flat_list = [item for sublist in img for item in sublist]
    return flat_list    

def min2d(img):   
    flat_list = flatten(img)  
    minElement= min1d(flat_list)
    return minElement   

def max2d(img):
    flat_list = flatten(img)
    maxElement = max1d(flat_list)
    return maxElement   


def normalize(img):
    '''
    normalize pixel intinsity range of an image to [0, 1]
    - findMin : loop over 2d array to get min element
    - findMax : loop over 2d array to get max element
    - sub     : element wise subtraction from 2d array 
    '''
    img_min = min2d(img)
    img_max = max2d(img)

    return  (sub2d(img, img_min)) / (img_max - img_min)   


def get_histogram(image, bins):
    image_flattened = flatten(image)
    # array with size of bins, set to zeros
    histogram = np.zeros(bins)
    # loop through pixels and sum up counts of pixels
    for pixel in image_flattened:
        histogram[pixel] += 1
    return histogram


def cumsum(hist):
    hist = iter(hist)
    b = [next(hist)]
    for i in hist:
        b.append(b[-1] + i)
    return np.array(b) 


def cumSumNorm(hist):
    cs = cumsum(hist)
    # re-normalize cumsum values to be between 0-255
    nj = (cs - min1d(cs)) * 255
    N = max1d(cs) - min1d(cs)
    # re-normalize the cdf
    cs = nj / N
    # cast it back to uint8 since we can't use floating point values in images
    cs_norm = cs.astype('uint8')
    return cs_norm


def histEqualize(image):
    flatten_image = flatten(image)
    hist = get_histogram(image, 256)
    # get the value from cumulative sum for every index in flat, and set that as img_new
    cs_norm = cumSumNorm(hist)
    equalized_image = cs_norm[flatten_image]
    return equalized_image




# seyam

def conv_step(img_slice , kernel, filter_type):
	''' 
	Args:
		img_slice: slice of image that will convolved with image
		kernel :   filter / mask 
		filter_type : type of filter

	Returns:
		convolved image
	'''

	if img_slice.shape != kernel.shape:
		print("the two shapes are different\n",img_slice, kernel)
		
	else :
		nrows,ncols = img_slice.shape
		result = np.zeros((nrows,ncols))
		for row in range(nrows):
			for col in range(ncols):
				result[row][col] = img_slice[row][col] * kernel[row][col]

		if filter_type == "average":
			sum_img  = img_sum(result)
			return sum_img/(kernel.shape[0]*kernel.shape[1])

		if filter_type == "median":
			med_img = img_median(result)
			return med_img

		if filter_type == "gaussian":
		    guass_img = img_sum(result)
		    return guass_img


def img_flatten(img):
	'''Args:
		img : matrix 

		Return:
		flattened_img : unrolled version of img matrix
	'''
	nrows, ncols = img.shape
	flattend_img = []

	for row in range(nrows):
			for col in range(ncols):
				flattend_img.append(img[row][col])
	
	return flattend_img



def img_sum(img):
	'''
		Args: 
			img (matrix)
		Return: 
			img_sum : value of summation
	'''
	nrows, ncols = img.shape
	img_sum = 0
	for row in range(nrows):
		for col in range(ncols):
			img_sum += img[row][col]
	
	return img_sum

def guassian_kernel(kernel_size, sigma):
	'''
		Args: 
		kernel_size: tuple of size of kernel
		sigma : the standard devation (more sigma -- more bluring)
		
		Return:
		kernel: auto generated kernel from normal distribution
	'''
	size,_ = kernel_size
	x0 = y0 = size // 2
	kernel = np.zeros((size, size))
	for i in range(size):
		for j in range(size):
			kernel[i, j] = np.exp(-(0.5/(sigma*sigma)) * (np.square(i-x0) + 
			np.square(j-y0))) / np.sqrt(2*(22/7)*sigma*sigma)

	kernel = kernel/img_sum(kernel)

	return kernel 


def img_median(img):
	''' Args:
			img: matrix

		Return:
			value : median value
	'''
	img = img_flatten(img)		# to unroll the matrix into list 
	sorted_image = sorted(img)
	img_size = len(sorted_image)
	
	if img_size % 2 == 0 :
		value =  (sorted_image[img_size/2 - 1] + sorted_image[img_size/2])/2
		return value

	if len(sorted_image)% 2 != 0 :
		value = sorted_image[img_size//2]
		return value 



def convolution(img, filter_type , filter_size, sigma = 1):

	''' 
	Args:
			img : the image that will be filtered (nd.array of int8)
			filter_type : just one of theree filters (Average , Guassian, Median filter)
			filter_size : the size of filter (tuple)
	Return:
			filtered_img : image after filteration (convolution)
	'''
	nrows, ncols = img.shape
	filter_height, filter_width = filter_size
	filter_img_width = ncols - filter_width + 1
	filter_img_height = nrows - filter_height + 1
	print(filter_img_height, filter_img_width)

	filtered_img = np.zeros((filter_img_height,filter_img_width)) # this is valid filter i think 
	
		
	for col in range(filter_img_width):
			horiz_start = col 
			horiz_end = col + filter_width 

			for row  in range(filter_img_height):
				vert_start = row 
				vert_end = row + filter_height

				img_slice = img[horiz_start:horiz_end, vert_start:vert_end]
				if filter_type in ["average","median"] :
					kernel = np.uint8(np.ones(filter_size))
				if filter_type == "gaussian":
					kernel = guassian_kernel(filter_size, sigma)

				update_val = conv_step(img_slice, kernel, filter_type) 
				filtered_img[col][row] = update_val


	return np.uint8(filtered_img)

# Galloul

def compare_after_noise(image, function):
	'''
    To compare between original image and the output image of a given function.

    Args:
        image (np.array): the original image before any change.
        function (fun): the function that returns the second image for comparison after some changes.
    '''

	fig, axs = plt.subplots(1, 2, figsize=(14, 7))
	axs[0].imshow(image, cmap="gray");
	axs[1].imshow(function(image.copy()), cmap="gray");


def add_uniform_noise(image, offset=0):
	'''
    Add uniform noise to a given image.

    Args:
        image (np.array): the image to add noise to.
        offset (int, default=0): how much noise to add to every pixel.
                                if not specified by thee user, it's randomly selected

    Returns:
        noised_image (np.array): the image after uniform noise was added.

    '''

	uniform_noise = np.ones_like(image, dtype=np.uint8)

	if (offset == 0):  # Generate random uniform noise
		offset = np.random.randint(0, 255);

	uniform_noise *= offset

	print("Uniform noise value is:", offset)

	noised_image = image + uniform_noise
	return noised_image


def add_saltNpepper_noise(image, Ws=0.1, Wp=0.1):
	'''
    Add salt&pepper noise to a given image and specify how much weight for every type.

    Args:
        image (np.array): the image to add noise to.
        Ws (float, default=0.1): the salt weight in the produced image.
        Wp (float, default=0.1): the pepper weight in the produced image.

    Returns:
        image (np.array): the image after salt&pepper noise was added.

    '''

	w, h = image.shape[0], image.shape[1]

	no_salted_pixels = int(w * h * Ws);
	no_peppered_pixels = int(w * h * Wp);

	for i in range(no_salted_pixels):
		image[np.random.randint(0, w), np.random.randint(0, h)] = 255;

	for i in range(no_peppered_pixels):
		image[np.random.randint(0, w), np.random.randint(0, h)] = 0;

	print(f" Adding noise with salt weight: {Ws} and pepper weight:{Wp}")
	return image


def add_gaussian_noise(image, mean=0, std=10):
	'''
    Add gaussian noise to a given image and specify the mean and std of the noise values driven from the guassian distribution.

    Args:
        image (np.array): the image to add noise to.
        mean (float, default=0): the mean of the gaussian distribution which the noise values are taken from.
        std (float, default=10): the standard deviation of the gaussian distribution which the noise values are taken from.

    Returns:
        noised_image (np.array): the image after gaussian noise was added.

    '''
	np.random.seed(int(time.time()))

	gaussian_noise = np.random.normal(mean, std, size=image.shape)

	print(f"Adding gaussian noise with mean={mean} and std={std}")

	noised_image = image + 2 * gaussian_noise
	return noised_image


def gradient_detector(image, detector_type='sobel'):
	'''
    Apply a gradient detector to a given image to extract edges in it.

    Args:
        image (np.array): the image to detect edges in.
        detector_type (str, default='sobel', options={'sobel', 'prewitt', 'roberts'}):
                        the type of kernel applied to extract edges.

    Returns:
        gradients_mag (np.array): the gradient magnitude for every pixel after applying the selected edge kernel,
                                    given by (sqrt(Ix^2 + Iy^2))
        gradients_angle (np.array): the gradient direction for every pixel after applying the selected edge kernel,
                                    calculated in rads.

    '''

	if detector_type == 'sobel':

		kernel_x = np.array([[-1, 0, 1],
							 [-2, 0, 2],
							 [-1, 0, 1]])

		kernel_y = kernel_x.T * -1  # equivelent to -> np.array([[ 1,  2,  1],
															 #   [ 0,  0,  0],
															 #   [-1, -2, -1]])

	elif detector_type == 'prewitt':
		kernel_x = np.array([[-1, 0, 1],
							 [-1, 0, 1],
							 [-1, 0, 1]])

		kernel_y = kernel_x.T * -1  # equivelent to -> np.array([[1, 1, 1],
	# [0, 0, 0],
	# [-1, -1, -1]])
	elif detector_type == 'roberts':

		kernel_x = np.array([[0, 1],
							 [-1, 0]])

		kernel_y = np.array([[1, 0],
							 [0, -1]])

	else:
		print("Unsupported detector, please choose either 'sobel', 'roberts', or 'prewitt' ")
		return None

	image_h, image_w = image.shape[0], image.shape[1]
	kernel_h, kernel_w = kernel_x.shape
	h, w = kernel_h // 2, kernel_w // 2

	gradients_x = np.zeros_like(image, dtype=np.uint8)
	gradients_y = np.zeros_like(image, dtype=np.uint8)
	gradients_mag = np.zeros_like(image, dtype=np.uint8)
	gradients_angle = np.zeros_like(image, dtype=np.uint8)

	for i in range(h, image_h - h):
		for j in range(w, image_w - w):
			conv_sum_x = 0
			conv_sum_y = 0

			for m in range(kernel_h):
				for n in range(kernel_w):
					conv_sum_x += kernel_x[m][n] * image[i - h + m][j - w + n]
					conv_sum_y += kernel_y[m][n] * image[i - h + m][j - w + n]

			gradients_x[i][j] = conv_sum_x
			gradients_y[i][j] = conv_sum_y
			gradients_mag[i][j] = (conv_sum_x ** 2 + conv_sum_y ** 2) ** 0.5

	gradients_angle = np.arctan2(gradients_y, gradients_x)

	return gradients_mag, gradients_angle


def non_max_suppression(gradient_mag, gradient_direction_rad):
	'''
    Apply non-maximum suppression for the gradients magnitude of an image using its gradients direction

    Args:
    gradients_mag (np.array): the gradient magnitude at every pixel, given by (sqrt(Ix^2 + Iy^2))
    gradient_direction_rad (np.array): the gradient direction at every pixel, calculated in rads.

    Returns:
        suppressed_img (np.array): the new image after suppression.

    '''
	h, w = gradient_mag.shape
	suppressed_img = np.zeros_like(gradient_mag, dtype=np.uint8)
	angle = gradient_direction_rad * 180. / np.pi
	angle[angle < 0] += 180  # ---> all angles are mapped between 0 & 180

	for i in range(1, h - 1):
		for j in range(1, w - 1):
			try:
				q = 255
				r = 255

				# angle 0
				if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
					q = gradient_mag[i, j + 1]
					r = gradient_mag[i, j - 1]
				# angle 45
				elif (22.5 <= angle[i, j] < 67.5):
					q = gradient_mag[i + 1, j - 1]
					r = gradient_mag[i - 1, j + 1]
				# angle 90
				elif (67.5 <= angle[i, j] < 112.5):
					q = gradient_mag[i + 1, j]
					r = gradient_mag[i - 1, j]
				# angle 135
				elif (112.5 <= angle[i, j] < 157.5):
					q = gradient_mag[i - 1, j - 1]
					r = gradient_mag[i + 1, j + 1]

				if (gradient_mag[i, j] >= q) and (gradient_mag[i, j] >= r):
					suppressed_img[i, j] = gradient_mag[i, j]
				else:
					suppressed_img[i, j] = 0

			except IndexError as e:
				pass

	return suppressed_img


def threshold(image, weak_pixel=75, strong_pixel=255, lowThresholdRatio=0.05, highThresholdRatio=0.2):
	'''
    Apply double-thresholding for a given image by checking whether the pixel value is strong, weak, or zero.

    Args:
        image (np.array): the image to detect apply double thresholding on.
        weak_pixel (int, default=75): the value which will be given for the weak pixel
        strong_pixel (int, default=255): the value which will be given for the strong pixel
        lowThresholdRatio (float, default=0.05): the min threshold to nejlect pixels under it as they are zero pixels
        highThresholdRatio (float, default=0.2): the min threshold to consider pixels above it as they are strong pixels
                                                the remaining pixels which are not consider either strong or zero pixels,
                                                are weak pixels and are assigned with the same weak value.

    Returns:
        result (np.array): the new image after double-thresholding was applied .

    '''

	highThreshold = image.max() * highThresholdRatio;
	lowThreshold = highThreshold * lowThresholdRatio;

	h, w = image.shape
	result = np.zeros_like(image, dtype=np.uint8)

	weak_pixel = np.uint8(weak_pixel)
	strong_pixel = np.uint8(strong_pixel)

	for i in range(h):
		for j in range(w):
			if (image[i, j] >= highThreshold):  # strong pixel
				result[i, j] = strong_pixel
			elif (image[i, j] < highThreshold and image[i, j] > lowThreshold):  # weak pixel
				result[i, j] = weak_pixel
			else:
				pass

	return result


def hysteresis(image, weak_value=75, strong_value=255):
	'''
    Apply hysteresis for a given image by checking if the weak pixels in it are have a strong adjacent pixel at least, hence
    consider it as a strong pixel assigned with the strong_value

    Args:
        image (np.array): the image to detect apply hysteresis on.
        weak_value (int, default=75): the value needed to consider a pixel weak.
        strong_value (int, default=255): the value given to the weak pixel if it has a single strong neighbor pixel at least

    Returns:
        image (np.array): the new image after hysteresis was applied .

    '''

	w, h = image.shape
	for i in range(1, w - 1):
		for j in range(1, h - 1):
			if (image[i, j] == weak_value):
				try:
					if ((image[i + 1, j - 1] == strong_value) or (image[i + 1, j] == strong_value)
							or (image[i + 1, j + 1] == strong_value) or (image[i, j - 1] == strong_value)
							or (image[i, j + 1] == strong_value) or (image[i - 1, j - 1] == strong_value)
							or (image[i - 1, j] == strong_value) or (image[i - 1, j + 1] == strong_value)):

						image[i, j] = strong_value

					else:
						image[i, j] = 0

				except IndexError as e:
					pass
	return image


def canny_detector(image, gradient_type='sobel', weak_pixel=50, strong_pixel=255, low_threshold_ratio=0.1,
				   high_threshold_ratio=0.5):
	'''
    Apply canny detector to detect edges in an image.

    Args:
        image (np.array): the image to detect edges in.
        detector_type (str, default='sobel', options={'sobel', 'prewitt', 'roberts'}):
                        the type of kernel applied to extract edges.

        weak_pixel (int, default=50): the value which will be given for the weak pixel
        strong_pixel (int, default=255): the value which will be given for the strong pixel
        low_threshold_ratio (float, default=0.1): the min threshold to nejlect pixels under it as they are zero pixels
        high_threshold_ratio (float, default=0.5): the min threshold to consider pixels above it as they are strong pixels
                                                    the remaining pixels which are not consider either strong or zero pixels,
                                                    are weak pixels and are assigned with the same weak value.



    Returns:
        final_image (np.array): the image to after edges were detected.

    '''

	# STEP1: apply gaussian blurring
	# image_blurred = cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)  # --> REPLACE WITH SEYAM'S FUNCTION FOR GAUSSIAN DENOISING
	image_blurred = convolution(image, "gaussian", (3, 3))
	# STEP2: apply gradient detector (e.g. sobel)
	gradient_mag, gradient_direction = gradient_detector(image_blurred, detector_type=gradient_type)

	# STEP3: apply non-maximum suppression to get thinner lines
	suppressed_image = non_max_suppression(gradient_mag, gradient_direction)

	# STEP4: apply double-thresholding to distinguish between weak, strong and irrelevant pixels
	threshold_image = threshold(suppressed_image, weak_pixel=weak_pixel, strong_pixel=strong_pixel,
								lowThresholdRatio=low_threshold_ratio, highThresholdRatio=high_threshold_ratio)

	# STEP5: apply hysteresis to make any weak pixel strong if any adjacent pixel is strong; zero otherwise.
	final_image = hysteresis(threshold_image, weak_value=weak_pixel, strong_value=strong_pixel)

	#     plt.imshow(final_image, cmap="gray")
	return final_image


def image_details(image):
	'''
    Returns imgae details whether if it is a RBG or a gray image

    Args:
        image (np.array): a gray or RGB image.

    Returns:
        w, h, d (int, int, int): the width, height and depth of the image.
        col_dict (dict): every channel found in the image and its color.
    '''
	try:  # RGB image
		w, h, d = image.shape
		col_dict = {0: 'r', 1: 'g', 2: 'b'}

	except:  # Gray-scale image
		w, h = image.shape
		d = 1
		col_dict = {0: 'gray'}

	return w, h, d, col_dict


def value_counts(channel_image):
	'''
    Returns the count of each pixel value in an image channel

    Args:
        channel_image (np.array): a 2d array of pixels.

    Returns:
        dic (dict): every pixel value found  and the corresponding count.
    '''

	dic = {}
	for value in channel_image:
		try:
			dic[value] += 1
		except:
			dic[value] = 1
	dic = dict(sorted(dic.items()))
	return dic


def draw_histogram_sns(image, plot_type='hist'):
	'''
    Returns a histogram/desity_distribution for every channel in a given image using seaborn built-in function.

    Args:
        image (np.array): the image to plot histogram for.
        plot_type (str, default='hist', options={'kde', 'hist'}): either a histogam for all the channels or the density curve.

    '''

	w, h, d, col_dict = image_details(image)

	reshaped_image = image.reshape(-1, d)

	df = pd.DataFrame(columns={'Pixel Intensity', 'color'})

	for i in range(d):
		df_ = pd.DataFrame(reshaped_image[:, i], columns={'Pixel Intensity'})
		df_['color'] = col_dict[i]
		df = pd.concat([df, df_], ignore_index=True)

	sns.displot(data=df, aspect=2, x='Pixel Intensity', hue='color', kind=plot_type, palette=list(col_dict.values()))


def draw_histogram_scratch(image, plot_type='hist'):
	'''
    Returns a histogram/desity_distribution for every channel in a given image.

    Args:
        image (np.array): the image to plot histogram for.
        plot_type (str, default='hist', options={'kde', 'hist'}): either a histogam for all the channels or the density curve.

    '''

	w, h, d, col_dict = image_details(image)
	cols = list(col_dict.values())
	reshaped_image = image.reshape(-1, d)

	figure = plt.figure(figsize=(14, 7));
	for i in range(d):
		dic = value_counts(reshaped_image[:, i])
		pixels, pixel_count = list(dic.keys()), list(dic.values())

		if plot_type == "hist":
			plt.bar(x=pixels, height=pixel_count, alpha=0.5, color=col_dict[i])
			plt.ylabel('Count')

		elif plot_type == "kde":
			normalized_pixel_count = [i / (w * h) for i in pixel_count]
			plt.plot(pixels, normalized_pixel_count, alpha=0.7, color=col_dict[i])
			plt.ylabel('Density')

		else:
			print("Unsupported plot type, please choose either 'hist' or 'kde'\t")
			return None

	legend = plt.legend(cols)
	legend = legend.set_title('color')

	plt.xlabel('Pixel Intensity')



##########################################
####### Problem 7 Global and Local #######
##########################################


threshold_values = {}
h = [1]


def Hist(img):
   ''' 
    Calculate histogram of the image with its plot.
   '''
   row, col = img.shape 
   y = np.zeros(256)
   for i in range(0,row):
      for j in range(0,col):
         y[img[i,j]] += 1
   x = np.arange(0,256)
   plt.bar(x, y, color='b', width=5, align='center', alpha=0.25)
   plt.show()
   return y


def binary_image(img, threshold):
    row, col = img.shape 
    y = np.zeros((row, col))
    for i in range(0,row):
        for j in range(0,col):
            if img[i,j] >= threshold:
                y[i,j] = 255
            else:
                y[i,j] = 0
    return y


   
def countPixel(h):
    cnt = 0
    for i in range(0, len(h)):
        if h[i]>0:
           cnt += h[i]
    return cnt


def wieght(s, e):
    w = 0
    for i in range(s, e):
        w += h[i]
    return w


def mean(s, e):
    m = 0
    w = wieght(s, e)
    for i in range(s, e):
        m += h[i] * i
    
    return m/float(w)


def variance(s, e):
    v = 0
    m = mean(s, e)
    w = wieght(s, e)
    for i in range(s, e):
        v += ((i - m) **2) * h[i]
    v /= w
    return v
            

def global_threshold(h):
    cnt = countPixel(h)
    for i in range(1, len(h)):
        vb = variance(0, i)
        wb = wieght(0, i) / float(cnt)
        mb = mean(0, i)
        
        vf = variance(i, len(h))
        wf = wieght(i, len(h)) / float(cnt)
        mf = mean(i, len(h))
        
        V2w = wb * (vb) + wf * (vf)      # minimize within class variance

        V2b = wb * wf * (mb - mf)**2   # maximize between class variance
        
        if not math.isnan(V2w):
            threshold_values[i] = V2w


def optimal_threshold_otsu():
    '''
    Otsu's method has two options 
        - The first is to minimize the within-class variance   [ V2w = wb * vb +  wf * vf ], where b,f are two classes 
        - The second is to maximize the between-class variance [ V2b = wb * wf * (mb - mf)**2] 
    1- calculate the histogram and intensity level probabilities [P(i) = n_i/n , where i gray-level value]
    2- initialize w_i, m_i.
    3- iterate over possible thresholds: t = 0 ==> max_intensity.
        - update the values of w_i, m_i, where w_i is a probability and m_i is a mean of class i
            [w_b(t) = sum( (P(i) ) from i=1 > i=t,
             w_f(t) = sum( (P(i) ) from i=t+1 > i=I]
        - calculate the within-class variance value V2w
    4- the final threshold is the minimum of V2w value.
    '''
    min_V2w = min(iter(threshold_values.values()))
    optimal_threshold = [k for k, v in iter(threshold_values.items()) if v == min_V2w]
    print('optimal threshold', optimal_threshold[0])
    return optimal_threshold[0]


def global_thresholding(image):
	'''
	global thresholding function that return binarized image.
	'''
	# important note: to make a function in a module use a global variable
	# use global of that variable inside a function to be  available in all scopes 
	global h
	global threshold_values

	img = np.asarray(image)
	h = Hist(img)
	global_threshold(h)
	op_thres = optimal_threshold_otsu()
	binary_img = binary_image(img, op_thres)
	return binary_img

def comparison_plot(img1, img2):
	plt.figure(figsize=(10, 14))
	plt.subplot(121)
	plt.imshow( img1 , cmap= 'gray')
	plt.subplot(122)
	plt.imshow( img2 , cmap='gray')
	plt.show()




def adaptative_thresholding(img, threshold=15, window_coef=16 ):
    if not  0 < threshold < 100:
        raise IOError('threshold must be between 0 and 100')

    n_rows, n_cols = img.shape
    
    # Windows size
    M = int(np.floor(n_rows/window_coef) + 1)
    N = int(np.floor(n_cols/window_coef) + 1)
    
    # Image border padding related to windows size
    Mextend = round(M/2)-1
    Nextend = round(N/2)-1
    
    # Padding image
    aux =cv2.copyMakeBorder(img, top=Mextend, bottom=Mextend, left=Nextend,
                          right=Nextend, borderType=cv2.BORDER_REFLECT)
    
    windows = np.zeros((M,N),np.int32)
    
    # Image integral calculation
    imageIntegral = cv2.integral(aux, windows,-1)
    
    # Integral image size
    nrows, ncols = imageIntegral.shape
    
    # Memory allocation for cumulative region image
    result = np.zeros((n_rows, n_cols))
    
    # Image cumulative pixels in windows size calculation
    for i in range(nrows-M):
        for j in range(ncols-N):
        
            result[i, j] = imageIntegral[i+M, j+N] - imageIntegral[i, j+N]+ imageIntegral[i, j] - imageIntegral[i+M,j]
     
    # Output binary image memory allocation    
    binar = np.ones((n_rows, n_cols), dtype=np.bool)
    
    # Gray image weighted by windows size
    graymult = (img).astype('float64')*M*N
    
    # Output image binarization
    binar[graymult <= result*(100.0 - threshold)/100.0] = False
    
    # binary image to UINT8 conversion
    binar = (255*binar).astype(np.uint8)
    
    return binar



def compare_three_imgs(img1, img2, img3):
	plt.figure(figsize=(14, 18))
	plt.subplot(131)
	plt.imshow(img1, cmap= 'gray')
	plt.title("Original")
	plt.subplot(132)
	plt.imshow(img2, cmap='gray')
	plt.title("Global")
	plt.subplot(133)
	plt.imshow(img3, cmap='gray')
	plt.title("Local")
	plt.show()

########################
#######	P8  ############
########################

def rgb2gray(rgb):
    '''
    For images in color spaces such as Y'UV which are used in standard color TV and video systems 
    luma component (Y') is calculated directly from gamma-compressed primary intensities as a weighted sum
     In the Y'UV and Y'IQ models used by PAL and NTSC  Y' is calclated as:
                    Y'= 0.299 R'+ 0.587 G'+ 0.114 B'
    '''
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])



################################
############# P9 ###############
################################

def omega(p, q):
   ''' The omega term in DFT and IDFT formulas'''
   return cmath.exp((2.0 * cmath.pi * 1j * q) / p)

def pad(lst):
   '''padding the list to next nearest power of 2 as FFT implemented is radix 2'''
   k = 0
   while 2**k < len(lst):
      k += 1
   return np.concatenate((lst, ([0] * (2 ** k - len(lst)))))

def fft(x):
   ''' FFT of 1-d signals
   usage : X = fft(x)
   where input x = list containing sequences of a discrete time signals
   and output X = dft of x '''

   n = len(x)
   if n == 1:
      return x
   Feven, Fodd = fft(x[0::2]), fft(x[1::2])
   combined = [0] * n
   for m in range(n//2):
     combined[m] = Feven[m] + omega(n, -m) * Fodd[m]
     combined[m + n//2] = Feven[m] - omega(n, -m) * Fodd[m]
   return combined

def ifft(X):
   ''' IFFT of 1-d signals
   usage x = ifft(X) 
   unpadding must be done implicitly'''

   x = fft([x.conjugate() for x in X])
   return [x.conjugate()/len(X) for x in x]

def pad2(x):
   m, n = np.shape(x)
   M, N = 2 ** int(ceil(log(m, 2))), 2 ** int(ceil(log(n, 2)))
   F = np.zeros((M,N), dtype = x.dtype)
   F[0:m, 0:n] = x
   return F, m, n

def fft2(f):
   '''FFT of 2-d signals/images with padding
   usage X, m, n = fft2(x), where m and n are dimensions of original signal'''

   f, m, n = pad2(f)
   return np.transpose(fft(np.transpose(fft(f)))), m, n

def ifft2(F, m, n):
   ''' IFFT of 2-d signals
   usage x = ifft2(X, m, n) with unpaded, 
   where m and n are odimensions of original signal before padding'''

   f, M, N = fft2(np.conj(F))
   f = np.matrix(np.real(np.conj(f)))/(M*N)
   return f[0:m, 0:n]

def fftshift(F):
   ''' this shifts the centre of FFT of images/2-d signals'''
   M, N = F.shape
   R1, R2 = F[0: M//2, 0: N//2], F[M//2: M, 0: N//2]
   R3, R4 = F[0: M//2, N//2: N], F[M//2: M, N//2: N]
   sF = np.zeros(F.shape,dtype = F.dtype)
   sF[M//2: M, N//2: N], sF[0: M//2, 0: N//2] = R1, R4
   sF[M//2: M, 0: N//2], sF[0: M//2, N//2: N]= R3, R2
   return sF



def gaussian(r2, std=1):
    """
    Sample one instance from gaussian distribution regarding
    given squared-distance:r2, standard-deviation:std and general-constant:k
	A sampled number obtained from gaussian
    """
    return np.exp(-r2/(2.*std**2)) / (2.*np.pi*std**2)
    
def make_gaussian(std=1, size=None):
    """
    Creates a gaussian kernel regarding given size and std.
    A gaussian kernel with size of (size*size)
    """
    if size is None:
        size = np.floor(6 * std)
        if size % 2 == 0:
            size = size - 1
        else:
            size= size - 2
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    # x, y = np.mgrid[0:4, 0:4]
    # x = 0 1 2 3 4   y =  0 0 0 0 0  sampling space = 4 . . . . .
    #     0 1 2 3 4        1 1 1 1 1                   3 . . . . .
    #     0 1 2 3 4        2 2 2 2 2                   2 . . . . .
    #     0 1 2 3 4        3 3 3 3 3                   1 . . . . .
    #     0 1 2 3 4        4 4 4 4 4                   0 . . . . .
    #                                                    0 1 2 3 4
    distance = x**2+ y**2
    kernel = gaussian(r2=distance, std=std)
    return kernel / kernel.sum()

def freq_filters(image, lpf, hpf):
	imgX = image.shape[0]//2
	imgY = image.shape[1]//2
	lpfX = lpf.shape[0]//2
	lpfY = lpf.shape[1]//2
	hpfX = hpf.shape[0]//2
	hpfY = hpf.shape[1]//2
	padX_lpf = imgX-lpfX
	padY_lpf = imgY-lpfY 
	padX_hpf = imgX-hpfX
	padY_hpf = imgY-hpfY

	plt.figure(figsize=(15, 20))

	# 2d padding   : np.pad(2dArray, (xPadLeft, xPadRight), (yPadUp, yPadDown), padding_values)

	lpf = np.pad(lpf, [ ( padX_lpf , padX_lpf ), ( padY_lpf , padY_lpf )], mode='constant', constant_values=0)
	hpf = np.pad(hpf, [ ( padX_hpf , padX_hpf ), ( padY_hpf , padY_hpf )], mode='constant', constant_values=lpf.max())

	width = 256
	height = 256
	dim = (width, height)
	
	# resize image
	resized_lpf = cv2.resize(lpf, dim, interpolation = cv2.INTER_AREA)
	resized_hpf = cv2.resize(hpf, dim, interpolation = cv2.INTER_AREA)
	return resized_lpf, resized_hpf
