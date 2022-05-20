from PIL import Image, ImageStat
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import ndimage
import math
from random import randint
import cv2
from PIL import Image
import time

class Kmeans():
    def __init__(self, img_path , n_clusters):
        self.im_path = img_path
        self.im = Image.open(img_path)
        self.img_width , self.img_height = self.im.size
        self.img_pixels = self.im.load()

        self.result = self.fit(n_clusters)
        
        


    def converged(self, centroids , old_centroids):
        '''This is function used to check convergence.
        Args:
            centroids : 3D array for centroids of each rgb channel
            old_centroids : 3D array for old centroids of each rgb channel

        Return:
            boolen value (true is convergence and false if not convergence)
        '''
        if len(old_centroids) == 0:
            return False
        if len(centroids) <= 5 :
            a = 1
        elif len(centroids) <= 10 :
            a = 2
        else:
            a = 4

        for i in range(0, len(centroids)):
            cent = centroids[i]
            old_cent = old_centroids[i]

            # basically if the old centroid == centroid it should be convergence and we used a threshold a 
            if ((int(old_cent[0]) - a) <= cent[0] <= (int(old_cent[0]) + a)) and ((int(old_cent[1]) - a) <= cent[1] <= (int(old_cent[1]) + a)) and ((int(old_cent[2]) - a) <= cent[2] <= (int(old_cent[2]) + a)):
                continue
            else:
                return False

        return True

    def getMin(self, pixel, centroids):
        '''
        This is function compute index of centroid closes to the pixel

        Args:
        pixel :  tuple of two value in x,y
        centroids : [(),(),()]
        
        Return: 
            minIndex: index of the centroid closest to the pixel
        '''
        
        minDist = np.inf
        minIndex = 0

        for i in range(0, len(centroids)):
            dist = np.sqrt(int((centroids[i][0] - pixel[0]))**2 + int((centroids[i][1] - pixel[1]))**2 + int((centroids[i][2] - pixel[2]))**2)
            if dist < minDist:
                minDist = dist
                minIndex = i

        return minIndex
    def assign_pixels(self,centroids):
        '''
        This function assign each pixel to the closest centroids

        Args:
            centroids: list of three tuples for three rgb channel [(R_centroid),(g_centroid),(b_centoids)]
        
        Returns:
            clusters: dict each pixel assigned to each pixel
        '''

        clusters = {}
        
        for x in range(0 , self.img_width):
            for y in range(0 , self.img_height):
                pixel = self.img_pixels[x,y]
                minIndex = self.getMin(pixel, centroids)

                try : 
                    clusters[minIndex].append(pixel)
                except KeyError:
                    clusters[minIndex] = [pixel]
        
        return clusters

    def adjust_centroids(self, clusters):
        '''
        This is funciton used to locate centorids based on mean value
        
        Args:
            clusters: dict contain assigned cluster for each pixel
        Returns:
            new_centroids: list of tuples containing new centroid

        '''
        new_centroids = []
        keys = sorted(clusters.keys())

        for key in keys:
            mean = np.mean(clusters[key], axis = 0)
            new = (int(mean[0]), int(mean[1]), int(mean[2]))
            # print(str(key) + ":" + str(new))
            new_centroids.append(new)

        return new_centroids

    def fit(self, someK):
        centroids = []
        old_centroids = [] 

        rgb_range = ImageStat.Stat(self.im).extrema
        i = 1 

        for k in range (0, someK):
            cent = self.img_pixels[np.random.randint(0, self.img_width), np.random.randint(0,self.img_height)]
            centroids.append(cent)

        print("Centroids intilized")
        print("###################")

        while not self.converged(centroids, old_centroids) and  i <= 20 :
            # print("iteration #" + str(i) )
            i += 1

            old_centroids = centroids
            clusters = self.assign_pixels(centroids)
            centroids = self.adjust_centroids(clusters)

        print("===========================================")
        print("Convergence Reached!")
        print(centroids)

        return centroids

    def draw(self):
        if self.result is None :
            print("please run algorithm")

        img_original = plt.imread(self.im_path)
        img_cp = np.full((self.img_height,self.img_width,3), 255, dtype=np.uint8)
        

        for x in range(self.img_width):
            for y in range(self.img_height):
                RGB_value = self.result[self.getMin(self.img_pixels[x, y], self.result)]
                
                img_cp[y,x] = RGB_value

        fig , axs = plt.subplots(1,2,figsize = (12,6))

        axs[0].imshow(img_original)
        axs[0].set_title("original image")


        axs[1].imshow(img_cp)
        axs[1].set_title("segmented image")



        plt.show()



def neigbours(point_pos, image):
    # note we have 8 neigbours at max around each pixel so we want to calculate their coordinate and their intensity 
    ncols, nrows = image.shape

    x_pos = point_pos[0]
    y_pos = point_pos[1]
    height, width = image.shape

    valid_neigbours = []   # coordinates of valid pixels
    intensity_values = []  # corresponding intensity values of valid pixels

    if (x_pos > ncols) | (y_pos > nrows) | (x_pos <0) | (y_pos <0) :
        return valid_neigbours , intensity_values

    #Top 
    if (y_pos + 1)  <  height :
        valid_neigbours.append((x_pos,y_pos + 1))
        intensity_values.append(image[x_pos, y_pos + 1])
        
        # top right
        if (x_pos + 1 ) <  width :
            valid_neigbours.append((x_pos+1, y_pos +1))
            intensity_values.append(image[x_pos+1, y_pos +1])
        
        # top left
        if (x_pos - 1 ) >=  0 :
            valid_neigbours.append((x_pos - 1, y_pos +1))
            intensity_values.append(image[x_pos - 1, y_pos +1])
        
    # bottom 
    if (y_pos - 1) >=  0 :
        valid_neigbours.append((x_pos,y_pos - 1))
        intensity_values.append(image[x_pos, y_pos - 1])
        
        # bottom right
        if (x_pos + 1 ) < width :
            valid_neigbours.append((x_pos+1, y_pos -1))
            intensity_values.append(image[x_pos+1, y_pos -1])
        
        # bottom left
        if (x_pos - 1 ) >=  0 :
            valid_neigbours.append((x_pos - 1, y_pos -1))
            intensity_values.append(image[x_pos - 1, y_pos -1])
    
    # Right 
    if (x_pos + 1) <  width :
        valid_neigbours.append((x_pos + 1, y_pos ))
        intensity_values.append(image[x_pos + 1, y_pos ])
    
    # Left 
    if (x_pos - 1) >=  0 :
        valid_neigbours.append((x_pos - 1, y_pos ))
        intensity_values.append(image[x_pos - 1, y_pos ])


    return valid_neigbours, intensity_values

     
def regionGrowing(image, std , seed):
    height, width = image.shape
    pixel_visited = np.zeros((height,width))

    region = np.copy(image)
    region[:,:] = 0 # black image

    region[seed[0], seed[1]] = 255
    pixels_to_be_visited = []  # queue

    pixels_to_be_visited.append(seed)

    while pixels_to_be_visited:
        
        pixel = pixels_to_be_visited.pop()

        valid_neigbours, intensity_values = neigbours(pixel, image)

        if (np.std(intensity_values) < std):
            region[pixel[0], pixel[1]] = 255 

            for neigbour in valid_neigbours : 
                if (pixel_visited[neigbour[0], neigbour[1]]) != 1 :
                    pixel_visited[neigbour[0], neigbour[1]] = 1 

                    pixels_to_be_visited.append(neigbour)
        else:
            for neigbour in valid_neigbours:
                pixel_visited[neigbour[0], neigbour[1]] = 1

    return region
    
    

class MeanShift():
    def __init__(self, img_path, mode = 2 , H = 90, Hr = 90 , Hs = 90, Iter = 100):



        self.img_path = img_path
        self.mode = mode

        # Mode = 1 indicates that thresholding should be done based on H
        # Mode = 2 indicates that thresholding should be done based on Hs and Hr
                
        # Method getNeighbors
        # It searches the entire Feature matrix to find
        # neighbors of a pixel
        # param--seed : Row of Feature matrix (a pixel)
        # param--matrix : Feature matrix
        # param--mode : mode=1 uses 1 value of threshold that is H
        #               mode=2 uses 2 values of thresholds
        #                      Hr for threshold in range domain
        #                      Hs for threshold in spatial domain
        # returns--neighbors : List of indices of F which are neighbors to the seed
        
        self.H = H 
        self.Hr = Hr
        self.Hs = Hs 
        self.Iter = Iter
        self.img = cv2.imread(self.img_path, cv2.IMREAD_COLOR)
        self.opImg = np.zeros(self.img.shape, np.uint8)

        self.performMeanShift(self.img)
        # Method performMeanShift
        # The heart of the code. This function performs the
        # mean shift discontinuity preserving filtering for an image
        # param--img : Image we wish to filter
    def performMeanShift(self,img):
        clusters = 0
        F = self.createFeatureMatrix(img)
        # Actual mean shift implementation
        # Iterate over our Feature matrix until it is exhausted
        while(len(F) > 0):
            # print ('remPixelsCount : ' + str(len(F)))
            # Generate a random index between 0 and Length of 
            # Feature matrix so that we can choose a random
            # Seed
            randomIndex = randint(0,len(F)-1)
            seed = F[randomIndex]
            # Cache the seed as our initial mean
            initialMean = seed
            # Group all the neighbors based on the threshold H
            # H can be a single value or two values or range and
            # spatial fomain
            neighbors = self.getNeighbors(seed,F,self.mode)
            # print('found neighbors :: '+str(len(neighbors)))
            # If we get only 1 neighbor, which is the pixel itself,
            # We can directly mark it in our output image without calculating the shift
            # This condition helps us speed up a bit if we come across regions of single
            # pixel
            if(len(neighbors) == 1):
                F=self.markPixels([randomIndex],initialMean,F,clusters)
                clusters+=1
                continue
            # If we have multiple pixels, calculate the mean of all the columns
            mean = self.calculateMean(neighbors,F)
            # Calculate mean shift based on the initial mean
            meanShift = abs(mean-initialMean)
            # If the mean is below an acceptable value (Iter),
            # then we are lucky to find a cluster
            # Else, we generate a random seed again
            if(np.mean(meanShift)<self.Iter):
                F = self.markPixels(neighbors,mean,F,clusters)
                clusters+=1
        return clusters




    def getNeighbors(self, seed,matrix,mode=1):
        neighbors = []
        snAppend = neighbors.append
        qrt = math.sqrt
        for i in range(0,len(matrix)):
            cPixel = matrix[i]
            # if mode is 1, we threshold using H
            if (mode == 1):
                d = qrt(sum((cPixel-seed)**2))
                if(d<self.H):
                    snAppend(i)
            # otherwise, we threshold using H
            else:
                r = qrt(sum((cPixel[:3]-seed[:3])**2))
                s = qrt(sum((cPixel[3:5]-seed[3:5])**2))
                if(s < self.Hs and r < self.Hr ):
                    snAppend(i)
        return neighbors

    # Method markPixels
    # Deletes the pixel from the Feature matrix
    # Marks the pixel in the output image with the mean intensity
    # param--neighbors : Indices of pixels (from F) to be marked
    # param--mean : Range and spatial properties for the pixel to be marked
    # param--matrix : Feature matrix
    # param--cluster : Cluster number

    def markPixels(self, neighbors,mean,matrix,cluster):
        for i in neighbors:
            cPixel = matrix[i]
            x=cPixel[3]
            y=cPixel[4]
            self.opImg[x][y] = np.array(mean[:3],np.uint8)
         
        return np.delete(matrix,neighbors,axis=0)

    # Method calculateMean
    # Calculates mean of all the neighbors and returns a 
    # mean vector
    # param--neighbors : List of indices of pixels (from F)
    # param--matrix : Feature matrix
    # returns--mean : Vector of mean of spatial and range properties
    def calculateMean(self, neighbors,matrix):
        neighbors = matrix[neighbors]
        r=neighbors[:,:1]
        g=neighbors[:,1:2]
        b=neighbors[:,2:3]
        x=neighbors[:,3:4]
        y=neighbors[:,4:5]
        mean = np.array([np.mean(r),np.mean(g),np.mean(b),np.mean(x),np.mean(y)])
        return mean

    # Method createFeatureMatrix
    # Creates a Feature matrix of the image 
    # as list of [r,g,b,x,y] for each pixel
    # param--img : Image for which we wish to comute Feature matrix
    # return--F : Feature matrix
    def createFeatureMatrix(self, img):
        h,w,d = img.shape
        F = []
        FAppend = F.append
        for row in range(0,h):
            for col in range(0,w):
                r,g,b = img[row][col]
                FAppend([r,g,b,row,col])
        F = np.array(F)
        return F
    def draw(self):
        
        fig, axs = plt.subplots(1,2, figsize = (12,10))
        
        axs[0].imshow(self.img)
        axs[0].set_title('original image')

        axs[1].imshow(self.opImg)
        axs[1].set_title('opt image')
        



def euclidean_distance(point1, point2):
    """Computes euclidean distance of point1(list) and point2(list).
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def clusters_distance(cluster1, cluster2):
    """Computes distance between two clusters.
    cluster is lists of lists of points
    """
    return max([euclidean_distance(point1, point2) for point1 in cluster1 for point2 in cluster2])
  
def clusters_distance_2(cluster1, cluster2):
    """Computes distance between two centroids of the two clusters
    """
    cluster1_center = np.average(cluster1, axis=0)
    cluster2_center = np.average(cluster2, axis=0)
    return euclidean_distance(cluster1_center, cluster2_center)


class AgglomerativeClustering:
    
    def __init__(self, k=2, initial_k=25):
        self.k = k
        self.initial_k = initial_k
        
    def initial_clusters(self, points):
        """partition pixels into self.initial_k groups based on color similarity
        """
        groups = {}
        d = int(256 / (self.initial_k))
        for i in range(self.initial_k):
            j = i * d
            groups[(j, j, j)] = []
        for i, p in enumerate(points):
            go = min(groups.keys(), key=lambda c: euclidean_distance(p, c))  
            groups[go].append(p)
        return [g for g in groups.values() if len(g) > 0]
        
    def fit(self, points):
        # initially, assign each point to a distinct cluster
        print('Computing initial clusters ...')
        self.clusters_list = self.initial_clusters(points)
        # print('number of initial clusters:', len(self.clusters_list))
        print('merging clusters ...')
        
        while len(self.clusters_list) > self.k:
            # Find the closest (most similar) pair of clusters
            cluster1, cluster2 = min([(c1, c2) for i, c1 in enumerate(self.clusters_list) for c2 in self.clusters_list[:i]],
                 key=lambda c: clusters_distance_2(c[0], c[1]))
            # Remove the two clusters from the clusters list
            self.clusters_list = [c for c in self.clusters_list if c != cluster1 and c != cluster2]
            # Merge the two clusters
            merged_cluster = cluster1 + cluster2
            # Add the merged cluster to the clusters list
            self.clusters_list.append(merged_cluster)        
        print('assigning cluster num to each point ...')
        self.cluster = {}
        for cl_num, cl in enumerate(self.clusters_list):
            for point in cl:
                self.cluster[tuple(point)] = cl_num
                
        print('Computing cluster centers ...')
        self.centers = {}
        for cl_num, cl in enumerate(self.clusters_list):
            self.centers[cl_num] = np.average(cl, axis=0)
                    
    def predict_cluster(self, point):
        """Find cluster number of point"""
        # assuming point belongs to clusters that were computed by fit functions
        return self.cluster[tuple(point)]

    def predict_center(self, point):
        """Find center of the cluster that point belongs to"""
        point_cluster_num = self.predict_cluster(point)
        center = self.centers[point_cluster_num]
        return center



def agglomerate_clustering(img, pixels, n_clusters):
    '''cluster the input image using agglomerative clustering'''
    tic = time.time()
    agglo = AgglomerativeClustering(k=n_clusters, initial_k=25)
    agglo.fit(pixels)

    new_img = [[agglo.predict_center(pixel) for pixel in row] for row in img]
    new_img = np.array(new_img, np.uint8)
    toc = time.time()
    print(f"time for k={n_clusters} clusters: {np.round(toc-tic,2)}")

    clustered_img = new_img
    return clustered_img    
