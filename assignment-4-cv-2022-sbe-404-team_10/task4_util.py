'''test_util.py
contain helper functions used in the task.
'''
##########################################
############## Imports ###################
##########################################

import numpy as np
import cv2

import numpy as np
import matplotlib.pyplot as plt

import math
import cv2

from itertools import combinations
import numpy as np


##########################################
############ Global and Local ############
##########################################


threshold_values = {}
h = [1]


def Hist(img):
   '''Calculate histogram of the image with its plot.'''
   row, col = img.shape 
   y = np.zeros(256)
   for i in range(0,row):
      for j in range(0,col):
         y[img[i,j]] += 1
   x = np.arange(0,256)
#    plt.bar(x, y, color='b', width=5, align='center', alpha=0.25)
#    plt.show()
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
    
    return 
    -----
    optimal_threshold:   int
    '''
    min_V2w = min(iter(threshold_values.values()))
    optimal_threshold = [k for k, v in iter(threshold_values.items()) if v == min_V2w]
    print('optimal threshold', optimal_threshold[0])
    return optimal_threshold[0]


def global_thresholding(image):
	'''
	global thresholding function that return binarized image.

    input
    -----
    image: pixel values
    
    output
    ------
    binary_image: the binarized image after otsu
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




def adaptative_thresholding(img, threshold=15, window_coef=16 ):
    '''
    Adaptive thresholding is a form of thresholding that takes into account spatial variations in illumination. 
    using the integral image of the input is fast method to perform local thresholding. Integral image solution
    is more robust to illumination changes in the image.

    inputs
    ------
    img        : grey-scale image. 
    threshold  : threshold to consider pixel as foreground or background.
    window_coef: the size of the sub-regions, will affect thickness of the binarized output.

    outputs
    -------
    binar:  binary image
    '''
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


def comparison_plot(img1, img2):
	plt.figure(figsize=(10, 14))
	plt.subplot(121)
	plt.imshow( img1 , cmap= 'gray')
	plt.subplot(122)
	plt.imshow( img2 , cmap='gray')
	plt.show()

def compare_three_imgs(img1, img2, img3, title1="img1", title2="img2", title3="img3"):
	plt.figure(figsize=(14, 18))
	plt.subplot(131)
	plt.imshow(img1, cmap= 'gray')
	plt.title(title1)
	plt.subplot(132)
	plt.imshow(img2, cmap='gray')
	plt.title(title2)
	plt.subplot(133)
	plt.imshow(img3, cmap='gray')
	plt.title(title3)
	plt.show()

#######################################################
############ Optimal - Otsu - Spectral ################
############      global - local       ################
#######################################################

def convert_binary(image_matrix, thresh_val):
    white = 255
    black = 0
    
    initial_conv = np.where((image_matrix <= thresh_val), image_matrix, white)
    final_conv = np.where((initial_conv > thresh_val), initial_conv, black)
    
    return final_conv


def otsu_threshold(image=None, hist=None):
    """ Find otsu optimal threshold.
    The otsu threshold is calculate with maximizing 
    between class variance depending on the CDF from
    the image histogram.
    [can also be calculated by finding threshold value
    that minimize within class variance].
    
    inputs
    ------
    image: The input image (ndarray)
    hist: The input image histogram (ndarray)

    outputs
    -------
    The Otsu threshold (int)
    """
    if image is None and hist is None:
        print('You must pass as a parameter either the input image or its histogram')

    # Calculating histogram
    if not hist:
        hist = np.histogram(image, bins=range(256))[0].astype(np.float)

    cdf_background = np.cumsum(np.arange(len(hist)) * hist)
    w_background = np.cumsum(hist)  # The number of background pixels
    w_background[w_background == 0] = 1  # To avoid divisions by zero
    m_backg = cdf_background / w_background  # The means

    cdf_foreground = cdf_background[-1] - cdf_background
    w_foreground = w_background[-1] - w_background  # The number of foreground pixels
    w_foreground[w_foreground == 0] = 1  # To avoid divisions by zero
    m_foreg = cdf_foreground / w_foreground  # The means

    var_between_classes = w_background * w_foreground * (m_backg - m_foreg) ** 2

    return np.argmax(var_between_classes)   

    
def _get_variance(hist, n_hist, cdf, thresholds):
    """Get the total variance of regions for a given set of thresholds"""
    variance = 0
    for i in range(len(thresholds) - 1):
        # Thresholds
        t1 = thresholds[i] + 1
        t2 = thresholds[i + 1]
        # Cumulative histogram
        weight = n_hist[t2] - n_hist[t1 - 1]
        # Region CDF
        r_cdf = cdf[t2] - cdf[t1 - 1]
        # Region mean
        r_mean = r_cdf / weight if weight != 0 else 0
        variance += weight * r_mean ** 2
    return variance

def _get_thresholds(hist, n_hist, cdf, nthrs):
    """Get the thresholds that maximize the variance between regions
    inputs
    ------
    hist: The histogram of the image (ndarray)
    n_hist: The normalized histogram of the image (ndarray)
    cdf: The cummulative distribution function of the histogram (ndarray)
    nthrs: The number of thresholds (int)
    outputs
    -------
    opt_thresholds: thresholds that maximize the variance between regions
    """
    # Thresholds combinations
    thr_combinations = combinations(range(255), nthrs)
    max_var = 0
    opt_thresholds = None
    # Extending histograms for convenience
    n_hist = np.append(n_hist, [0])
    cdf = np.append(cdf, [0])

    for thresholds in thr_combinations:
        # Extending thresholds for convenience
        e_thresholds = [-1]
        e_thresholds.extend(thresholds)
        e_thresholds.extend([len(hist) - 1])
        # Computing variance for the current combination of thresholds
        regions_var = _get_variance(hist, n_hist, cdf, e_thresholds)
        if regions_var > max_var:
            max_var = regions_var
            opt_thresholds = thresholds
    return opt_thresholds


def otsu_multithreshold(image=None, hist=None, nthrs=2):
    """ find otsu's multi-thresholds values.
    inputs
    ------
    image: The input image (ndarray)
    hist: The histogram of the image (ndarray)
    nthrs: The number of thresholds (int)
    outputs
    -------
    optimal thresholds that maximize the variance between regions.
    """
    # Histogran
    if image is None and hist is None:
        raise ValueError('You must pass as a parameter either'
                         'the input image or its histogram')

    # Calculating histogram
    if not hist:
        hist = np.histogram(image, bins=range(256))[0].astype(np.float)

    # Cumulative histograms
    c_hist = np.cumsum(hist)
    cdf = np.cumsum(np.arange(len(hist)) * hist)

    return _get_thresholds(hist, c_hist, cdf, nthrs)


def plot_multi_level_otsu(image, th_list, colormap="gray"):
    # Using the threshold values, we generate the three regions.
    regions = np.digitize(image, th_list)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))
    # Plotting the original image.
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')
    # Plotting the histogram and the two thresholds obtained from
    # multi-Otsu.
    ax[1].hist(image.ravel(), bins=255)
    ax[1].set_title('Histogram')
    for thresh in th_list:
        ax[1].axvline(thresh, color='r')
    # Plotting the Multi Otsu result.
    ax[2].imshow(regions, cmap=colormap)
    ax[2].set_title('Multi-Otsu result')
    ax[2].axis('off')

    plt.subplots_adjust()

    plt.show()


def localOtsuThresholding_optimized(image,window_size):
    # match image dim to [ max dimension] as block size is square!
    img_dim = image.shape
    max_dim = np.max(img_dim)
    resizedImage = cv2.resize(image, (max_dim, max_dim))
    rows = resizedImage.shape[0]
    cols = resizedImage.shape[1]

    thresholdedImage = np.zeros(resizedImage.shape)

    for r in range(0, rows, window_size):
        for c in range(0, cols, window_size):
            # Extarct blocks `min(r+window_size, rows)` ensure that the block isn't outside the image
            block = resizedImage[r:min(r + window_size, rows), c:min(c + window_size, cols)]
            # use from scratch otsu_threshold
            otsuThreshold = otsu_threshold(block)
            # convert to binary [ (0 , 255) only]
            thresholdedBlock = convert_binary(block, otsuThreshold)
            # fill the output image for each block
            thresholdedImage[r:min(r + window_size, rows), c:min(c + window_size, cols)] = thresholdedBlock

    # resize output  image back to original size
    thresholdedImage = cv2.resize(thresholdedImage, (image.shape[1], image.shape[0]))
    return thresholdedImage




def local_spectral_multilevel(image,window_size, num_th ):
    # match image dim to [ max dimension] as block size is square!
    img_dim = image.shape
    max_dim = np.max(img_dim)
    resizedImage = cv2.resize(image, (max_dim, max_dim))
    rows = resizedImage.shape[0]
    cols = resizedImage.shape[1]

    thresholdedImage = np.zeros(resizedImage.shape)

    for r in range(0, rows, window_size):
        for c in range(0, cols, window_size):
            # Extarct blocks `min(r+window_size, rows)` ensure that the block isn't outside the image
            block = resizedImage[r:min(r + window_size, rows), c:min(c + window_size, cols)]
            # use from scratch otsu_threshold
            # otsuThreshold = otsu_threshold(block)
            otsuThreshold_list = otsu_multithreshold(block, nthrs=num_th)
            # convert to binary [ (0 , 255) only]
            thresholdedBlock =  np.digitize(block, otsuThreshold_list)
            # fill the output image for each block
            thresholdedImage[r:min(r + window_size, rows), c:min(c + window_size, cols)] = thresholdedBlock

    # resize output  image back to original size
    thresholdedImage = cv2.resize(thresholdedImage, (image.shape[1], image.shape[0]))
    return thresholdedImage, otsuThreshold_list


def plot_local_spectral_multilevel(image,digitized_img,otsuThreshold_list, colormap="gray"):
    # Using the threshold values, we generate the three regions.
    # regions = np.digitize(image, th_list)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))
    # Plotting the original image.
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')
    # Plotting the histogram and the two thresholds obtained from
    # multi-Otsu.
    ax[1].hist(image.ravel(), bins=255)
    ax[1].set_title('Histogram')
    for thresh in otsuThreshold_list:
        ax[1].axvline(thresh, color='r')
    # Plotting the Multi Otsu result.
    ax[2].imshow(digitized_img, cmap=colormap)
    ax[2].set_title('Multi-Otsu result')
    ax[2].axis('off')

    plt.subplots_adjust()

    plt.show()




def global_Optimal_Thresholding(image):
    rows = image.shape[0]
    cols = image.shape[1]
    # get  initial background mean  (4corners)
    background = [image[0, 0], image[0, cols-1], image[rows-1, 0], image[rows-1, cols-1]]
    background_mean = np.mean(background)
    # get  initial foreground mean
    foreground_mean = np.mean(image) - background_mean
    # get  initial threshold
    thresh = (background_mean + foreground_mean) / 2.0

    while True:
        old_thresh = thresh
        new_foreground = image[np.where(image >= thresh)]
        new_background = image[np.where(image < thresh)]
        # print(new_background.size)
        if new_background.size:
            new_background_mean = np.mean(new_background)
        else:
            new_background_mean = 0
        if new_foreground.size:
            new_foreground_mean = np.mean(new_foreground)
        else:
            new_foreground_mean = 0
        # update threshold
        thresh = (new_background_mean + new_foreground_mean) / 2
        if old_thresh == thresh:
            break
    return round(thresh, 2)


def local_Optimal_Thresholding(image, block_size):
    # blockSize = 16
    img_dim = image.shape
    max_dim = np.max(img_dim)
    resizedImage = cv2.resize(image, (max_dim, max_dim))
    
    rows = resizedImage.shape[0]
    cols = resizedImage.shape[1]
    outputImage = np.zeros(resizedImage.shape)

    for r in range(0, rows, block_size):
        for c in range(0, cols, block_size):
            # Extarct blocks
            block = resizedImage[r:min(r + block_size,rows), c:min(c + block_size, cols)]
            # get  initial background mean  (4corners)
            thresh = global_Optimal_Thresholding(block)
            # convert to binary [ (0 , 255) only]
            thresholdedBlock = convert_binary(block, thresh)
            # fill the output image for each block
            outputImage[r:min(r + block_size, rows), c:min(c + block_size, cols)] = thresholdedBlock

    # resize output  image back to original size
    outputImage = cv2.resize(outputImage, (image.shape[1], image.shape[0]))

    return outputImage





