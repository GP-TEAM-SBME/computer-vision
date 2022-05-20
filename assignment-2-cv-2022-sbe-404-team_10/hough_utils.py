import numpy as np 
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cv2 
import imageio



def hough_line_accumulator(img, rho_resolution = 1, theta_resolution = 1):
	''' A function create hough accumulator for line in image.
	Args:
		 img: edge img in image space resulted from canny detector. 
		 rho_resolution : resolution in rho direction in hough space. 
		 theta_resolution : resolution in theta direction in hough space.

	Return:
		  H_accumulator: accumulator array in hough space 
		  thetes:  array containing all theta values
		  rhos : values of rohs in hough space 
	'''


	# get height and width of original image to calculate image diagonal
	height, width = img.shape 
	img_diagonal = np.ceil(np.sqrt(height ** 2 + width **2))

	# put range of values of rho and theta to create hough space 
	rhos = np.arange(-img_diagonal , img_diagonal , rho_resolution)
	thetas = np.deg2rad(np.arange(-90,90,theta_resolution))

	# create an empty hough accumulator with size = len of rhos and thetas
	H_accumulator =  np.zeros((len(rhos), len(thetas)), dtype = np.uint64)

	# find all edge pixel (non zero pixels (white pixels))
	y_idxs, x_idxs = np.nonzero(img)

	for i in range(len(x_idxs)):
		x = x_idxs[i]
		y = y_idxs[i]

		for j in range(len(thetas)):  # for each edge pixel 
			# Calculate rho. diag_len is added for a positive index
			rho = int(x * np.cos(thetas[j]) + y * np.sin(thetas[j]) + img_diagonal) # compute rho = xcos(theta) + y sin(theta)

			# voting 
			H_accumulator[rho,j] += 1   

	return  H_accumulator, thetas, rhos 


def hough_simple_peaks(H, num_peaks):
	''' A function that returns the number of indicies = num_peaks of the
		accumulator array H that correspond to local maxima. 
		
		return indices of rhos and thetas of maximum element.
		'''
	indices =  np.argpartition(H.flatten(), -2)[-num_peaks:]
	return np.vstack(np.unravel_index(indices, H.shape)).T


def plot_hough_accumulator(H_accumulator,  plot_title = "Hough accumlator plot"):
	''' A function that plot a Hough Space using Matplotlib. '''
	fig = plt.figure(figsize=(10, 10))
	fig.canvas.set_window_title(plot_title)
		
	plt.imshow(H_accumulator, cmap='jet')

	plt.xlabel('Theta Direction'), plt.ylabel('Rho Direction')
	plt.tight_layout()
	plt.show()
# drawing the lines from the Hough Accumulatorlines using OpevCV cv2.line

def hough_lines_draw(img, indicies, rhos, thetas):
    ''' A function that takes indicies a rhos table and thetas table and draws
        lines on the input images that correspond to these values. '''

    for i in range(len(indicies)):
        # reverse engineer lines from rhos and thetas
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho

        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(img, pt1, pt2, (255,0,0), 1)



def detectCircles(img,threshold,region,radius = None):
    ''' this is function used to detect circles in images and return circle coordinates
    Args: img : edge images resulted from canny 
          threshold: threshold value for detecting circle 
          radius : range of radius available in picture
          
    
    Returns:
        coordinate of circles detected in image
    '''

    (M,N) = img.shape
    
    if radius == None:
        R_max = np.max((M,N))
        R_min = 3
    else:
        [R_max,R_min] = radius

  
    #Initializing accumulator array.
    #Accumulator array is a 3 dimensional array with the dimensions representing
    #the radius, X coordinate and Y coordinate resectively.
    #Also appending a padding of 2 times R_max to overcome the problems of overflow
    A = np.zeros((R_max,M+2*R_max,N+2*R_max))      # for points at boundary to draw circle without overflow
    B = np.zeros((R_max,M+2*R_max,N+2*R_max))

    #Precomputing all angles to increase the speed of the algorithm
    theta = np.arange(0,360)*np.pi/180
    edges = np.argwhere(img[:,:])                                               #Extracting all edge coordinates
    for r in range(R_min , R_max):


        #Creating a Circle Blueprint
        bprint = np.zeros((2*(r+1),2*(r+1)))
        (m,n) = (r+1,r+1)   # center pixel                                              
        for angle in theta:
            x = int(np.round(r*np.cos(angle)))
            y = int(np.round(r*np.sin(angle)))
            bprint[m+x,n+y] = 1
        constant = np.argwhere(bprint).shape[0]
        for x,y in edges:                                                       #For each edge coordinates
            
         
            A[r,x-m+R_max:x+m+R_max,y-m+R_max:y+m+R_max] += bprint
        A[r][A[r]<threshold*constant/r] = 0

    for r,x,y in np.argwhere(A):
        temp = A[r-region:r+region,x-region:x+region,y-region:y+region]
        try:
            p,a,b = np.unravel_index(np.argmax(temp),temp.shape)
        except:
            continue
        B[r+(p-region),x+(a-region),y+(b-region)] = 1

    res = B[:,R_max:-R_max,R_max:-R_max] 

    return np.argwhere(res)


def displayCircles(A, file_path):
   
    img = plt.imread(file_path)
   
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img)
    
    for r,x,y in A:
        circ = Circle( (y,x), r, color=(1,0,0), fill=False )
        ax.add_patch(circ)

    
    print('finished....')
    plt.show()
