from itertools import cycle

import numpy as np
from scipy import ndimage as ndi
from matplotlib import pyplot as plt

from contour_util import gaussian_fft_filter, gradient_detector_azoz

class _fcycle(object):

    def __init__(self, iterable):
        """Call functions from the iterable each time it is called."""
        self.funcs = cycle(iterable)

    def __call__(self, *args, **kwargs):
        f = next(self.funcs)
        return f(*args, **kwargs)


# SI and IS operators for 2D images
#  0  1  0    0  0  0    1  0  0    0  0  1     
#  0  1  0    1  1  1    0  1  0    0  1  0
#  0  1  0    0  0  0    0  0  1    1  0  0

_P2 = [np.eye(3),
       np.array([[0, 1, 0]] * 3),
       np.flipud(np.eye(3)),
       np.rot90([[0, 1, 0]] * 3)]

def visualize_contour_segment(image, levelSet):

    fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image, cmap=plt.cm.gray)
    ax2 = fig.add_subplot(1, 2, 2)
    ax_u = ax2.imshow(np.zeros_like(image), vmin=0, vmax=1,  cmap=plt.cm.gray)

    if ax1.collections:
        del ax1.collections[0]
    ax1.contour(levelSet, [0.5], colors='r')
    ax_u.set_data(levelSet)
    fig.canvas.draw()


def sup_inf(u):
    '''
    SI operator
    u : level set
    '''

    if np.ndim(u) == 2:
        P = _P2
    else:
        print("u has an invalid number of dimensions should be 2d")

    erosions = []
    for P_i in P:
        erosions.append(ndi.binary_erosion(u, P_i))

    return np.array(erosions, dtype=np.int8).max(0)


def inf_sup(u):
    '''
    IS operator
    u : level set
    '''
    if np.ndim(u) == 2:
        P = _P2
    else:
        print("u has an invalid number of dimensions should be 2d")
    dilations = []
    for P_i in P:
        dilations.append(ndi.binary_dilation(u, P_i))

    return np.array(dilations, dtype=np.int8).min(0)


_curvop = _fcycle([lambda u: sup_inf(inf_sup(u)),   # SIoIS
                   lambda u: inf_sup(sup_inf(u))])  # ISoSI


def _check_input(image, init_level_set):
    """Check that shapes of `image` and `init_level_set` match."""
    if image.ndim != 2:
       print("image must be a 2-dimensional array")

    if len(image.shape) != len(init_level_set.shape):
        print("The dimensions of the initial level set do not match the dimensions of the image")



def circle_level_set(image_shape, center=None, radius=None):
    '''Create a circle level set with binary values.
    Parameters
    ----------
    center : optional, If not given it defaults to the center of the image.
    radius : optional, If not given, it is set to the 75% of the smallest image dimension.
    
    Returns
    -------
    out : Binary level set of the circle with the given `radius` and `center`.
    '''

    if center is None:
        center = tuple(i // 2 for i in image_shape)

    if radius is None:
        radius = min(image_shape) * 3.0 / 8.0

    grid = np.mgrid[[slice(i) for i in image_shape]]
    grid = (grid.T - center).T
    phi = radius - np.sqrt(np.sum((grid)**2, 0))
    res = np.int8(phi > 0)
    return res

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

def morphological_geodesic_active_contour(gimage, iterations,
                                          circle_level_set, smoothing=1,
                                          threshold='auto', balloon=0):
                                        #   , iter_callback=lambda x: None
    '''Morphological Geodesic Active Contours (MorphGAC).
    Geodesic active contours implemented with morphological operators. It can
    be used to segment objects with visible but noisy, cluttered, broken borders.
    Parameters
    ----------
    gimage : Preprocessed image to be segmented that enhances and highlights the borders of the object.
            `Morph-GAC` will try to stop the contour evolution in areas where `gimage` is small.
    iterations : Number of iterations to run.
    init_level_set : Initial level set.
    smoothing : Number of times the smoothing operator is applied per iteration.
    threshold : Areas of the image with a value smaller than this threshold will be
                considered borders. 
    balloon :  Balloon force to guide the contour in areas where the gradient of the image is too small 
               A negative value will shrink the contour, a positive value will expand the contour.
    Returns
    -------
    out :      Final level set
    '''
    image = gimage
    init_level_set = circle_level_set

    _check_input(image, init_level_set)

    if threshold == 'auto':
        threshold = np.percentile(image, 40)

    structure = np.ones((3,) * len(image.shape), dtype=np.int8)
    dimage = np.gradient(image)
    # threshold_mask = image > threshold
    if balloon != 0:
        threshold_mask_balloon = image > threshold / np.abs(balloon)

    u = np.int8(init_level_set > 0)

    for _ in range(iterations):
        # Balloon
        if balloon > 0:
            aux = ndi.binary_dilation(u, structure)
        elif balloon < 0:
            aux = ndi.binary_erosion(u, structure)
        if balloon != 0:
            u[threshold_mask_balloon] = aux[threshold_mask_balloon]

        # Image attachment
        aux = np.zeros_like(image)
        du = np.gradient(u)
        for el1, el2 in zip(dimage, du):
            aux += el1 * el2
        u[aux > 0] = 1
        u[aux < 0] = 0

        # Smoothing
        for _ in range(smoothing):
            u = _curvop(u)
        
    return u