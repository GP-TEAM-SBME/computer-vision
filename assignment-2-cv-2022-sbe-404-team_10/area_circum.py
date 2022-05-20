import numpy as np
from scipy import ndimage
import cv2
from contour_util import convolve2D


def boundary_erosion(binary_img, structure_element=3):
    '''
    src: file:///C:/Users/Mohamed%20Abdelaziz/Downloads/perimeter.pdf
    A simple method for identifying the boundary pixels is to perform 
    an Erosion operation on the image.
    The boundary pixels are those which were eroded, and can be 
    found by subtracting the result from the original image.
    Parameters
    ----------
      binary_img : A black-and-white image
      structure_element : Kernal size for structuring element 
    Returns
    -------
      perim : A boolean image
    '''
    img_eroded = ndimage.binary_erosion(binary_img, structure=np.ones((structure_element,structure_element))).astype(binary_img.dtype)
    img_boundary = binary_img -  img_eroded
    return img_boundary


def overlay_contour(image, levelSet):
    '''
    Function to overlay the countor over the original image.
    Parameters
    ----------
      img : An RGB image
      Segment : Levelset from the morphSnake output 'the segmented area (by curve evolution)' 
    Returns
    -------
      overlayed_img : Original RGB image with red Countour
    '''
    img_resized  = cv2.resize(image, (512,512), interpolation = cv2.INTER_AREA)
    img_boundary = boundary_erosion(levelSet)
    img_resized[img_boundary == 1] = (255,0,0)
    overlayed_img = img_resized

    return overlayed_img


def bwperim(bw, n=4):
    """
    (scikit-image definition: approximates the contour as a line through the centers of border pixels using a 4-connectivity.)
    Find the perimeter of objects in binary images. A pixel is part of an object perimeter 
    if its value is one and there is at least one zero-valued pixel in its neighborhood.
    Connectivity=8 => more connections as Two adjoining pixels are part of the same object 
    if they are both on and are connected along the horizontal, vertical, or diagonal direction.
    Parameters
    ----------
      bw : A black-and-white image
      n : Connectivity. Must be 4 or 8 (default: 4)
    Returns
    -------
      perim : A boolean image
    """
    if n not in (4,8):
        print('bwperim: n must be 4 or 8')
    rows,cols = bw.shape

    # Translate image by one pixel in all directions
    north = np.zeros((rows,cols))
    south = np.zeros((rows,cols))
    west = np.zeros((rows,cols))
    east = np.zeros((rows,cols))
    north[:-1,:] = bw[1:,:]
    south[1:,:]  = bw[:-1,:]
    west[:,:-1]  = bw[:,1:]
    east[:,1:]   = bw[:,:-1]
    idx = (north == bw) & \
          (south == bw) & \
          (west  == bw) & \
          (east  == bw)
    if n == 8:
        north_east = np.zeros((rows, cols))
        north_west = np.zeros((rows, cols))
        south_east = np.zeros((rows, cols))
        south_west = np.zeros((rows, cols))
        north_east[:-1, 1:]   = bw[1:, :-1]
        north_west[:-1, :-1] = bw[1:, 1:]
        south_east[1:, 1:]     = bw[:-1, :-1]
        south_west[1:, :-1]   = bw[:-1, 1:]
        idx &= (north_east == bw) & \
               (south_east == bw) & \
               (south_west == bw) & \
               (north_west == bw)
    return ~idx * bw

def calculate_peremiter(binary_img, connectivity=4):
    '''
    src: 'file:///C:/Users/Mohamed%20Abdelaziz/Downloads/perimeter.pdf'
    calculate an accurate perimeter by weighting pixels contribution from 3 categories 
    convolve with np.array([[10, 2, 10], [2, 1, 2],[10, 2, 10]])
    The result of this convolution at each pixel position enables the category of the corresponding 
    edge pixel to be deduced
    [(a): 5 or 15 or 7 or 25 or 27 or 17 , 
     (b): 13 or 23, 
     (c): 21 or 33]
    Parameter
    ---------
    binary_img  : Levelset(segment)
    connectivity: 4 for horizontal, vertical connectivity. 8 for diagonal direction. [bwperim : default= 4]
    Output
    ------
    perimeter: scalar No. of (a) pixels * 1  + No. of (b) pixels* $\sqrt 2$ + No. of (c) pixels * (1+  $\sqrt 2$)/2
    '''
    _a = [5, 15, 7, 25, 27, 17]
    _c = [13, 23]
    _b = [21, 33]
    
    kernel = np.array([[10, 2, 10], [2, 1, 2],[10, 2, 10]])
    # Convolve and Save Output
    perimeter_boundary = bwperim(binary_img, connectivity)
    # print("num of pixels by connectivity 4:  ", len(np.argwhere(perimeter_boundary == 1)))
    output = convolve2D(perimeter_boundary, kernel, padding=1)
    
    a  =  [len(np.argwhere(output == x)) for x in _a]
    b  =  [len(np.argwhere(output == x)) for x in _b]
    c  =  [len(np.argwhere(output == x)) for x in _c]

    # print('No. of (a) pixels : ', np.sum(a))
    # print('No. of (b) pixels : ', np.sum(b))
    # print('No. of (c) pixels : ', np.sum(c))

    # print("total a, b, c pixels :", np.sum(a)+np.sum(b)+np.sum(c))
    perimeter = ( np.sum(a)*1 ) + ( np.sum(b)*np.sqrt(2) ) + (np.sum(c)*(1+np.sqrt(2))/2)
    return perimeter


    
def bwarea(bw):
    '''
    bwarea estimates the area of all of the on pixels in an image by summing the areas of each pixel in the image.
    The area of an individual pixel is determined by looking at its 2-by-2 neighborhood. There are five different patterns,
    each representing a different area:
    - Patterns with one on pixel (area = 1/4)
    - Patterns with two adjacent on pixels (area = 1/2)
    - Patterns with two diagonal on pixels (area = 3/4)
    - Patterns with three on pixels (area = 7/8)
    - Patterns with all four on pixels (area = 1)
    parameter
    ---------
    bw: binary image
    output
    ------
    area: weighted sum of pixels area
    '''
    four = np.ones((2, 2))
    two = np.diag([1, 1])
    fours = convolve2D(bw, four)
    twos = convolve2D(bw, two)
    nQ1 = np.sum(fours == 1)
    nQ3 = np.sum(fours == 3)
    nQ4 = np.sum(fours == 4)
    nQD = np.sum(np.logical_and(fours == 2, twos != 1))
    nQ2 = np.sum(np.logical_and(fours == 2, twos == 1))

    area = 0.25 * nQ1 + 0.5 * nQ2 + 0.875 * nQ3 + nQ4 + 0.75 * nQD

    return area