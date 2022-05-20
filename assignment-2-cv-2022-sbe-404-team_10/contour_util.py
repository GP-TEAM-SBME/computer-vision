from turtle import width
import numpy as np
import matplotlib.pyplot as plt
from math import log, ceil
import cv2
import cmath

def rgb2gray(rgb):
    '''
    For images in color spaces such as Y'UV which are used in standard color TV and video systems 
    luma component (Y') is calculated directly from gamma-compressed primary intensities as a weighted sum
     In the Y'UV and Y'IQ models used by PAL and NTSC  Y' is calclated as:
                    Y'= 0.299 R'+ 0.587 G'+ 0.114 B'
    '''
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def comparison_plot(img1, img2):
   plt.figure(figsize=(10, 14))
   plt.subplot(121)
   plt.imshow( img1 , cmap= 'gray')
   plt.subplot(122)
   plt.imshow( img2 , cmap='gray')
   plt.show()




################################
############# FFT ###############
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
   # print(F.shape)
   M, N = F.shape
   R1, R2 = F[0: M//2, 0: N//2], F[M//2: M, 0: N//2]
   R3, R4 = F[0: M//2, N//2: N], F[M//2: M, N//2: N]
   sF = np.zeros(F.shape,dtype = F.dtype)
   sF[M//2: M, N//2: N], sF[0: M//2, 0: N//2] = R1, R4
   sF[M//2: M, 0: N//2], sF[0: M//2, N//2: N]= R3, R2
   # print("fft-shift", sF.shape)
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

   # plt.figure(figsize=(15, 20))

   # 2d padding   : np.pad(2dArray, (xPadLeft, xPadRight), (yPadUp, yPadDown), padding_values)

   lpf = np.pad(lpf, [ ( padX_lpf , padX_lpf ), ( padY_lpf , padY_lpf )], mode='constant', constant_values=0)
   hpf = np.pad(hpf, [ ( padX_hpf , padX_hpf ), ( padY_hpf , padY_hpf )], mode='constant', constant_values=lpf.max())

   width  = image.shape[0] 
   height = image.shape[1]
   dim = (height, width)
   
   # resize image
   resized_lpf = cv2.resize(lpf, dim, interpolation = cv2.INTER_AREA)
   resized_hpf = cv2.resize(hpf, dim, interpolation = cv2.INTER_AREA)
   return resized_lpf, resized_hpf






def gradient_detector_azoz(image):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2],[-1, 0, 1]])
    kernel_y = kernel_x.T * -1  
    
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


def image_gradient(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = convolve2D(img, Kx, padding=1)
    Iy = convolve2D(img, Ky, padding=1)
    
    G = np.hypot(Ix, Iy)
   #  G = G / G.max() * 255

    return G

def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
      #   print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output    




def gaussian_fft_filter(image, cutoff):

   if image.ndim == 3:
      img_gray = rgb2gray(image)
   else:
      img_gray = image

   # resized_img = cv2.resize(img_gray, (256,256))
   
   cutoff_lpf = cutoff  # >> 6*50 -1 = 299  >> resized to target img
   lpf = make_gaussian(cutoff_lpf)
   hpf = lpf.max() - lpf
   resized_lpf, resized_hpf= freq_filters(img_gray, lpf, hpf)

   # print(img_gray.shape)
   img_fft ,m1,n1 = fft2(img_gray) 
   img_fftshift = fftshift(img_fft)

   # width  = image.shape[0] 
   # height = image.shape[1]
   # dim = (height, width)
   # img_fftshift = np.asarray(img_fftshift)
   # img_fftshift = cv2.resize(img_fftshift, dim) 
   
   # print(img_fftshift.shape)
   # print(resized_lpf.shape)
   # w = 512
   # h = 512
   # dim = (w, h)
   # resized_lpf  = cv2.resize(lpf, dim, interpolation = cv2.INTER_AREA)
   # resized_hpf  = cv2.resize(hpf, dim, interpolation = cv2.INTER_AREA)

   img_lpf = ifft2(np.multiply(img_fftshift, resized_lpf),m1, n1)
   img_hpf = ifft2(np.multiply(img_fftshift, resized_hpf),m1, n1)

   # w = image.shape[0]
   # h = image.shape[1]
   # dim = (w, h)
   # img_lpf  = cv2.resize(img_lpf, dim, interpolation = cv2.INTER_AREA)
   # img_hpf  = cv2.resize(img_hpf, dim, interpolation = cv2.INTER_AREA)


   return  np.asarray(np.abs(img_lpf)*255) ,  np.asarray(np.abs(img_hpf)*255)


def inverse_gaussian_gradient(image, alpha=100, cutoff_lpf=20):
   '''
   a preprocessed version of the original image that enhances and highlights the borders 
   (or other structures) of the object to segment. `morphological_geodesic_active_contour`
   will try to stop the contour evolution in areas where `gimage` is small.  

   Parameters
   ----------
   image      : Grayscale image or RGB images 
   alpha      : Controls the steepness of the inversion.
   cutoff_lpf : Gaussian kernal size using fft freq_filters

   Returns
   -------
   gimage     : Preprocessed image suitable for `morphological_geodesic_active_contour`.
   '''
   image_lpf, _ = gaussian_fft_filter(image, cutoff_lpf)
   mag, _ = gradient_detector_azoz(image_lpf)
   gimage = 1.0 / np.sqrt(1.0 + alpha * mag)

   return gimage