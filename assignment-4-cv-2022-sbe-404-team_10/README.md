[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=7830560&assignment_repo_type=AssignmentRepo)
# Task 4 (Segmentation)

## 1. Local and Global Thresholding low level implementations

### 1.1 Global
Global Thresholding is using only one threshold value to binarize image into foreground and background.
### Global thresholding performs badly With Shadows and low contrast regions
Global thresholding is good when foreground and background has clear diffrence; hence low contrast regions, homogenous regions, and shadows are limitations of global thresholding. 



```python
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from task4_util import global_thresholding, adaptative_thresholding, comparison_plot, compare_three_imgs
```


```python
# This implementation uses low level functions implemented from scratch without numpy
image = Image.open('./images/avatar.png').convert("L")
binary_image = global_thresholding(image)
comparison_plot(image, binary_image)
```

    e:\SBME\2022-2ndtern\cv\tasks\task1-python\assignment-4-cv-2022-sbe-404-team_10\task4_util.py:77: RuntimeWarning: invalid value encountered in double_scalars
      return m/float(w)
    

    optimal threshold 106
    


    
![png](readme_notebook_files/readme_notebook_2_2.png)
    


### 1.2 Local Thresholding
We used Bradley implementation for local adaptive image thresholding based on integral image from 
[Bradley Paper Link](https://people.scs.carleton.ca/~roth/iit-publications-iti/docs/gerh-50002.pdf)

Adaptive thresholding is a form of thresholding that takes into account spatial variations in illumination. 
using the integral image of the input is fast method to perform local thresholding. Integral image solution
is more robust to illumination changes in the image. 

one limitation is the size of neighbourhood should be large enough to accomodate suffecient foreground and background.



```python
image = Image.open('./images/text.png').convert("L")
img = np.asarray(image)

global_th = global_thresholding(image)
local_th = adaptative_thresholding(img, 20)

compare_three_imgs(image, global_th, local_th, "Original", "Global", "Local [integral image]")

```

    optimal threshold 139
    


    
![png](readme_notebook_files/readme_notebook_4_1.png)
    



```python
image = Image.open('../images/veins.png').convert("L")
img = np.asarray(image)

global_th = global_thresholding(image)
local_th = adaptative_thresholding(img, 3, 8)

compare_three_imgs(image, global_th, local_th, "Original", "Global", "Local [integral image]")
```

    e:\SBME\2022-2ndtern\cv\tasks\task1-python\assignment-4-cv-2022-sbe-404-team_10\test\test_util.py:76: RuntimeWarning: invalid value encountered in double_scalars
      return m/float(w)
    

    optimal threshold 79
    


    
![png](readme_notebook_files/readme_notebook_5_2.png)
    



```python
image = Image.open('./images/jet.png').convert("L")
img = np.asarray(image)

tic = time.time()
global_th = global_thresholding(image)
toc = time.time()
print("global time", np.round(toc-tic,2))
tic = time.time()
local_th = adaptative_thresholding(img, 3, 8)
toc = time.time()
print("integral time", np.round(toc-tic,2))
compare_three_imgs(image, global_th, local_th, "Original", "Global", "Local [integral image]")
```

    optimal threshold 157
    global time 0.77
    integral time 0.26
    


    
![png](readme_notebook_files/readme_notebook_6_1.png)
    


### 2. Varient to from scratch otsu using numpy. and varient to local thresholding using integral image.

#### 2.1  global otsu  [[Otsu Paper]](https://www.semanticscholar.org/paper/A-threshold-selection-method-from-gray-level-Otsu/1d4816c612e38dac86f2149af667a5581686cdef)
#### 2.2  local  otsu   [`on sub-regions`]
#### 2.3  spectral global otsu  [`2 or more thresholds`]  
 - multiple regions in single channel [MULTI-LEVEL Thresholding Paper](http://smile.ee.ncku.edu.tw/old/Links/MTable/ResearchPaper/papers/2001/A%20fast%20algorithm%20for%20multilevel%20%20thresholding.pdf)
- threshold on spectral image channels (e.g., rgb)   [MULTISPECTRAL THRESHOLDING paper](https://www.researchgate.net/publication/306351469_EDGE_DETECTION_USING_MULTISPECTRAL_THRESHOLDING) as it is not required in our task we separate it to spectral.ipynb
#### 2.4  spectral local otsu  [`2 or more thresholds on image sub-regions`]  
#### 2.5  iterative optimal global thresholding 
#### 2.6  iterative optimal local thresholding



```python
from task4_util import convert_binary, otsu_threshold, otsu_multithreshold, plot_multi_level_otsu, localOtsuThresholding_optimized
```

## 2.1 Global OTSU


```python
image = Image.open('./images/avatar.png').convert("L")
image = np.asarray(image)

th = otsu_threshold(image)
print("otsu threshold: ", th)

binarized_img = convert_binary(image, th)
plt.imshow(binarized_img, cmap="gray")

```

    otsu threshold:  105
    


    <matplotlib.image.AxesImage at 0x2af544187f0>



    
![png](readme_notebook_files/readme_notebook_10_2.png)
    


## 2.2 Local OTSU
This is just using global otsu on  image sub-regions.

The drawbacks of this method is that the local neighbourhood should not be:
- small: as it will take very long time to compute and the threshold will be easily affected with noise.
- large: as large region will be more affected with illumination and shadows. 

**A major issue with local otsu is not the `time`, but the real issue is the `window edges` which need some interpolation method to result in more meaneangful binary out**

**This can be seen in the below images as the black patches, which is only deacreased by using smaller window**

> The window size of choice should be smaller than the segmented object which explain why the text binarization is behaving badly.
>
> The window showld be large enough to contain pixel values from the classes to have good segmentation. **[it will fail if the window contain one class]**

`Time on text.png`
|Type|time(text.png)|time optimized (text.png)|time(jet.png)| time optimized (jet.png)
|--------|------|-------|-----|------|
|global otsu| < .1 s| < .1 s | .7 s| .7 s |
|local [integral image]|  .1 s|  .1 s| .2 s| .2 s|
|local otsu (window 16)| 39 s |  .07 s| 86.7 s|   0.18 s|
|local otsu (window 32)| 10 s | .02 s| 23.6 s|   0.05 s|
|local otsu (window 64)|  3 s |  .01 s| 6.4 s|   0.02 s|
|local otsu (window 128)|  1 s| < .01 s| 1.8 s|   0.01 s|


>  For text image window 8x8 achieved segmentation similar to local segmentation using integral image in only .29 seconds which is very good result.


```python
windows  = [8, 32]
def plot_local(img):
    plt.imshow(img, cmap="gray")
    plt.show()
for window in windows:
    image = Image.open(f'./images/text.png').convert("L")
    image = np.asarray(image)
    tic = time.time()
    local_th_image = localOtsuThresholding_optimized(image, window)
    toc = time.time()
    plot_local(local_th_image)
    print(f"time for {window}x{window} window: {np.round(toc - tic, 2)}")

```


    
![png](readme_notebook_files/readme_notebook_12_0.png)
    


    time for 8x8 window: 0.27
    


    
![png](readme_notebook_files/readme_notebook_12_2.png)
    


    time for 32x32 window: 0.02
    


```python
windows  = [16, 32]
def plot_local(img):
    plt.imshow(img, cmap="gray")
    plt.show()
for window in windows:
    image = Image.open(f'./images/jet.png').convert("L")
    image = np.asarray(image)
    tic = time.time()
    local_th_image = localOtsuThresholding_optimized(image, window)
    toc = time.time()
    plot_local(local_th_image)
    print(f"time for {window}x{window} window: {np.round(toc - tic, 2)}")

```


    
![png](readme_notebook_files/readme_notebook_13_0.png)
    


    time for 16x16 window: 0.16
    


    
![png](readme_notebook_files/readme_notebook_13_2.png)
    


    time for 32x32 window: 0.06
    

## 2.3  Spectral (Multi-Level) Global OTSU  


```python
level_list = [1,2,3]
for level in level_list: 
    image = Image.open('./images/jet.png').convert("L")
    image = np.asarray(image) 
    tic = time.time()
    th_list = otsu_multithreshold(image, nthrs=level)
    print("otsu multi thresholds: ", th_list)
    plot_multi_level_otsu(image, th_list)    
    toc = time.time()
    print(f"time for multilevel otsu ({level} levels) : {np.round(toc-tic, 2)}")
```

    otsu multi thresholds:  (156,)
    


    
![png](readme_notebook_files/readme_notebook_15_1.png)
    


    time for multilevel otsu (1 levels) : 0.45
    otsu multi thresholds:  (115, 173)
    


    
![png](readme_notebook_files/readme_notebook_15_3.png)
    


    time for multilevel otsu (2 levels) : 0.71
    otsu multi thresholds:  (96, 147, 189)
    


    
![png](readme_notebook_files/readme_notebook_15_5.png)
    


    time for multilevel otsu (3 levels) : 20.2
    

## Segmenting multiple medical images using multi level otsu gives very good results in diffrent domains 
- Lung X-Ray
- Swallowing Fluroscopy
- Braim MRI 


```python
imgs_list = ["lung.jpg"]
level = 2
for img in imgs_list:
    image = Image.open(f'./images/{img}').convert("L")
    image = np.asarray(image) 
    tic = time.time()
    th_list = otsu_multithreshold(image, nthrs=2)
    plot_multi_level_otsu(image, th_list)    
    toc = time.time()
    print("otsu multi thresholds: ", th_list)
    print(f"time for multilevel otsu ({level} levels) : {np.round(toc-tic, 2)}")


```


    
![png](readme_notebook_files/readme_notebook_17_0.png)
    


    otsu multi thresholds:  (94, 152)
    time for multilevel otsu (2 levels) : 1.52
    


```python
imgs_list = ["ns050a_1.png", "ns050a_2.png", "brain.jpeg", "brain2.jpeg"]
for img in imgs_list:
    image = Image.open(f'./images/{img}').convert("L")
    image = np.asarray(image) 
    tic = time.time()
    th_list = otsu_multithreshold(image, nthrs=3)
    plot_multi_level_otsu(image, th_list)    
    toc = time.time()
    print("otsu multi thresholds: ", th_list)
    print(f"time for multilevel otsu ({level} levels) : {np.round(toc-tic, 2)}")
    

```


    
![png](readme_notebook_files/readme_notebook_18_0.png)
    


    otsu multi thresholds:  (61, 108, 179)
    time for multilevel otsu (3 levels) : 20.1
    


    
![png](readme_notebook_files/readme_notebook_18_2.png)
    


    otsu multi thresholds:  (62, 110, 179)
    time for multilevel otsu (3 levels) : 20.02
    


    
![png](readme_notebook_files/readme_notebook_18_4.png)
    


    otsu multi thresholds:  (33, 87, 163)
    time for multilevel otsu (3 levels) : 19.91
    


    
![png](readme_notebook_files/readme_notebook_18_6.png)
    


    otsu multi thresholds:  (23, 71, 116)
    time for multilevel otsu (3 levels) : 20.09
    

## 2.4  Spectral (Multi-Level) Local OTSU  


```python
from task4_util import local_spectral_multilevel, plot_local_spectral_multilevel
```


```python

imgs_list = ["jet.png"]
for img in imgs_list:
    image = Image.open(f'./images/{img}').convert("L")
    image = np.asarray(image) 
    tic = time.time()
    digitized_img, th_list = local_spectral_multilevel(image, 64 , 2)
    # plot_multi_level_otsu(image, th_list)  
    plot_local_spectral_multilevel(image, digitized_img, th_list)  
    toc = time.time()
    print("otsu multi thresholds: ", th_list)
    print(f"time for multilevel otsu (2 levels) : {np.round(toc-tic, 2)}")




```


    
![png](readme_notebook_files/readme_notebook_21_0.png)
    


    otsu multi thresholds:  (140, 177)
    time for multilevel otsu (2 levels) : 14.85
    


```python

imgs_list = ["lena.png"]
for img in imgs_list:
    image = Image.open(f'./images/{img}').convert("L")
    image = np.asarray(image) 
    tic = time.time()
    digitized_img, th_list = local_spectral_multilevel(image, 64 , 1)
    # plot_multi_level_otsu(image, th_list)  
    plot_local_spectral_multilevel(image, digitized_img, th_list)  
    toc = time.time()
    print("otsu multi thresholds: ", th_list)
    print(f"time for multilevel otsu (2 levels) : {np.round(toc-tic, 2)}")

```


    
![png](readme_notebook_files/readme_notebook_22_0.png)
    


    otsu multi thresholds:  (89,)
    time for multilevel otsu (2 levels) : 0.57
    

## 2.5 Global Iterartive OPtimal Thresholding
Global and Local `iterative optimal thresholding` reached the same result of otsu, but in slower time
|image|global_otsu|global_optimal
|--------|-----------| ---------|
|avatar.png| .02s | .15 s
|lung.jpg| .22 s | 1.63 s|



```python
from task4_util import global_Optimal_Thresholding, local_Optimal_Thresholding
```


```python
image = Image.open('./images/lung.jpg').convert("L")
image = np.asarray(image)

tic = time.time()
otsu_th = otsu_threshold(image)
toc = time.time()
otsu_img = convert_binary(image, otsu_th)
plt.imshow(otsu_img, cmap="gray")
plt.show()
print("otsu threshold: ", otsu_th)
print(f"global otsu time: {np.round(toc-tic, 2)}")

tic = time.time()
opt_th = global_Optimal_Thresholding(image)
toc = time.time()
iter_optimal_img = convert_binary(image, opt_th)
plt.imshow(iter_optimal_img, cmap="gray")
plt.show()
print("global iterative optimal threshold: ", otsu_th)
print(f"global iterative optimal time: {np.round(toc-tic, 2)}")

```


    
![png](readme_notebook_files/readme_notebook_25_0.png)
    


    otsu threshold:  119
    global otsu time: 0.25
    


    
![png](readme_notebook_files/readme_notebook_25_2.png)
    


    global iterative optimal threshold:  119
    global iterative optimal time: 1.61
    


```python
image = Image.open('./images/avatar.png').convert("L")
image = np.asarray(image)

tic = time.time()
otsu_th = otsu_threshold(image)
toc = time.time()
otsu_img = convert_binary(image, otsu_th)
plt.imshow(otsu_img, cmap="gray")
plt.show()
print("otsu threshold: ", otsu_th)
print(f"global otsu time: {np.round(toc-tic, 2)}")

tic = time.time()
opt_th = global_Optimal_Thresholding(image)
toc = time.time()
iter_optimal_img = convert_binary(image, opt_th)
plt.imshow(iter_optimal_img, cmap="gray")
plt.show()
print("global iterative optimal threshold: ", otsu_th)
print(f"global iterative optimal time: {np.round(toc-tic, 2)}")

```


    
![png](readme_notebook_files/readme_notebook_26_0.png)
    


    otsu threshold:  105
    global otsu time: 0.01
    


    
![png](readme_notebook_files/readme_notebook_26_2.png)
    


    global iterative optimal threshold:  105
    global iterative optimal time: 0.12
    

## 2.6 Local Iterative Optimal Thresholding

very similar results as local otsu, but slower 

|image|local_otsu|local_optimal|
|--------|--------|-------|
|jet.png| .17 s (window 16x16) & .05 s (window 32x32)|.39 s (window 8x8 ) &  .11 s (window 16x16)|
|text.png| .28 s (window 8x8) & .02 s (window 32x32) | .06 s (window 32x32 ) &  .04 s (window 64x64)|




```python
windows  = [8, 16]
def plot_local(img):
    plt.imshow(img, cmap="gray")
    plt.show()
for window in windows:
    image = Image.open(f'./images/text.png').convert("L")
    image = np.asarray(image)
    tic = time.time()
    local_th_image = localOtsuThresholding_optimized(image, window)
    toc = time.time()
    plot_local(local_th_image)
    print(f"time for {window}x{window} window: {np.round(toc - tic, 2)}")
```


    
![png](readme_notebook_files/readme_notebook_28_0.png)
    


    time for 8x8 window: 0.62
    


    
![png](readme_notebook_files/readme_notebook_28_2.png)
    


    time for 16x16 window: 0.11
    


```python
windows  = [32,64]
def plot_local(img):
    plt.imshow(img, cmap="gray")
    plt.show()
for window in windows:
    image = Image.open(f'./images/jet.png').convert("L")
    image = np.asarray(image)
    tic = time.time()
    local_th_image = local_Optimal_Thresholding(image, window)
    toc = time.time()
    plot_local(local_th_image)
    print(f"time for {window}x{window} window: {np.round(toc - tic, 2)}")

```


    
![png](readme_notebook_files/readme_notebook_29_0.png)
    


    time for 32x32 window: 0.05
    


    
![png](readme_notebook_files/readme_notebook_29_2.png)
    


    time for 64x64 window: 0.04
    

# <i> image segmentation

<p> image Segmentation involves converting an image into a collection of regions of pixels that are represented by a mask or a labeled image. By dividing an image into segments, you can process only the important segments of the image instead of processing the entire image.

<p> A common technique is to look for abrupt discontinuities in pixel values, which typically indicate edges that define a region.

<p> Another common approach is to detect similarities in the regions of an image. Some techniques that follow this approach are region growing, clustering, and thresholding.

<p> A variety of other approaches to perform image segmentation have been developed over the years using domain-specific knowledge to effectively solve segmentation problems in specific application areas.


## <i> image segmentation with Kmeans

<p> K-Means clustering algorithm is an unsupervised algorithm and it is used to segment the interest area from the background. It clusters, or partitions the given data into K-clusters or parts based on the K-centroids.

<p> The algorithm is used when you have unlabeled data(i.e. data without defined categories or groups). The goal is to find certain groups based on some kind of similarity in the data with the number of groups represented by K.

### steps of Kmean algorithm


- Choose the number of clusters K.
- Select at random K points, the centroids(not necessarily from your dataset).
- Assign each data point to the closest centroid → that forms K clusters.
- Compute and place the new centroid of each cluster.
- Reassign each data point to the new closest centroid. If any reassignment . took place, go to step 4, otherwise, the model is ready.


```python
from PIL import Image, ImageStat
import numpy as np
import os
import matplotlib.pyplot as plt
from task4_util_segmentation import Kmeans, regionGrowing, MeanShift

```


```python
img_path = os.path.join(os.getcwd(),'img/test10.jpg')
model = Kmeans(img_path=img_path,n_clusters = 2)
model.draw()
```

    Centroids intilized
    ###################
    ===========================================
    Convergence Reached!
    [(230, 187, 186), (59, 43, 57)]
    


    
![png](readme_notebook_files/readme_notebook_34_1.png)
    



```python
img_path = os.path.join(os.getcwd(),'img/test09.jpg')
model = Kmeans(img_path=img_path,n_clusters = 4)
model.draw()
```

    Centroids intilized
    ###################
    ===========================================
    Convergence Reached!
    [(9, 155, 158), (196, 149, 116), (10, 52, 53), (78, 57, 47)]
    


    
![png](readme_notebook_files/readme_notebook_35_1.png)
    


## <i> image segmentation with region growing

<p> Region-growing methods rely mainly on the assumption that the neighboring pixels within one region have similar values. The common procedure is to compare one pixel with its neighbors. If a similarity criterion is satisfied, the pixel can be set to belong to the cluster as one or more of its neighbors. The selection of the similarity criterion is significant and the results are influenced by noise in all instances.

<p> This method takes a set of seeds as input along with the image. The seeds mark each of the objects to be segmented. The regions are iteratively grown by comparison of all unallocated neighboring pixels to the regions. The difference between a pixel’s intensity value and the region’s mean, is used as a measure of similarity. The pixel with the smallest difference measured in this way is assigned to the respective region. This process continues until all pixels are assigned to a region. Because seeded region growing requires seeds as additional input, the segmentation results are dependent on the choice of seeds, and noise in the image can cause the seeds to be poorly placed.


```python
import cv2
image_path = os.path.join(os.getcwd() , "img/brain-slice40.tiff")
image = cv2.imread(image_path,0)

```


```python
seed = (150, 150)
region = regionGrowing(image, 9.8, seed)
```


```python
im_h, im_w = image.shape
# Display images
fig, ax = plt.subplots(1, 2, figsize=(8, 8))
ax[0].imshow(region, cmap="gray")
ax[0].set_title('Region growing of brain ({} px, {} px)'.format(im_h, im_w))
ax[0].axis('off')

ax[1].imshow(image, cmap="gray")
ax[1].set_title('MRI brain image ({} px, {} px)'.format(im_h, im_w))
ax[1].axis('off')

plt.tight_layout()
plt.show()
```


    
![png](readme_notebook_files/readme_notebook_40_0.png)
    



```python
image_path = os.path.join(os.getcwd(), "img/Butterfly.jpg")
image = cv2.imread(image_path,0)
plt.imshow(image, cmap="gray")
```


    <matplotlib.image.AxesImage at 0x145804691c0>



    
![png](readme_notebook_files/readme_notebook_41_1.png)
    



```python
# image_path = os.path.join(os.getcwd() , "img/Butterfly.jpg")
# image = cv2.imread(image_path,0)

seed = (230, 280)
region = regionGrowing(image, 9.8, seed)

im_h, im_w = image.shape
fig, ax = plt.subplots(1, 2, figsize=(8, 8))
ax[0].imshow(region, cmap="gray")
ax[0].set_title('Region growing of brain ({} px, {} px)'.format(im_h, im_w))
ax[0].axis('off')
ax[1].imshow(image, cmap="gray")
ax[1].set_title('MRI brain image ({} px, {} px)'.format(im_h, im_w))
ax[1].axis('off')
plt.tight_layout()
plt.show()
```


    
![png](readme_notebook_files/readme_notebook_42_0.png)
    


## <i> image segmentation with mean shift clustering

<p> Mean shift clustering is a non-parametric technique for clustering, it isn’t require to specify the number of clusters. Also it is robust for outliers as clusters aren’t in spherical shape it takes a none-linear shape according to clustering procedure.


```python
imgPath = os.path.join(os.getcwd(),'img/test03.jpg')

mean_shift = MeanShift(imgPath)
mean_shift.draw()

```


    
![png](readme_notebook_files/readme_notebook_45_0.png)
    


## Agglomerative Clustering

The agglomerative clustering is the most common type of hierarchical clustering used to group objects in clusters based on their similarity. It’s also known as AGNES (Agglomerative Nesting). The algorithm starts by treating each object as a singleton cluster. Next, pairs of clusters are successively merged until all clusters have been merged into one big cluster containing all objects. The result is a tree-based representation of the objects, named dendrogram.

[Agglomerative Clustering Resource](https://www.datanovia.com/en/lessons/agglomerative-hierarchical-clustering/)


```python
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from task4_util_segmentation import agglomerate_clustering
from task4_util import comparison_plot
```


```python
img = mpimg.imread("./images/donut.jpeg")
pixels = img.reshape((-1,3))

n_clusters = range(2,5)
for k in n_clusters:
    print(f'Processing k={k}:')
    clustered_img = agglomerate_clustering(img, pixels, k)
    comparison_plot(img, clustered_img)
```

    Processing k=2:
    Computing initial clusters ...
    merging clusters ...
    assigning cluster num to each point ...
    Computing cluster centers ...
    time for k=2 clusters: 11.22
    


    
![png](readme_notebook_files/readme_notebook_48_1.png)
    


    Processing k=3:
    Computing initial clusters ...
    merging clusters ...
    assigning cluster num to each point ...
    Computing cluster centers ...
    time for k=3 clusters: 11.58
    


    
![png](readme_notebook_files/readme_notebook_48_3.png)
    


    Processing k=4:
    Computing initial clusters ...
    merging clusters ...
    assigning cluster num to each point ...
    Computing cluster centers ...
    time for k=4 clusters: 11.19
    


    
![png](readme_notebook_files/readme_notebook_48_5.png)
    



```python
img = mpimg.imread("./images/mama07ORI.bmp")
pixels = img.reshape((-1,3))

n_clusters = range(2,5)
for k in n_clusters:
    print(f'Processing k={k}:')
    clustered_img = agglomerate_clustering(img, pixels, k)
    comparison_plot(img, clustered_img)




```

    Processing k=2:
    Computing initial clusters ...
    merging clusters ...
    assigning cluster num to each point ...
    Computing cluster centers ...
    time for k=2 clusters: 19.6
    


    
![png](readme_notebook_files/readme_notebook_49_1.png)
    


    Processing k=3:
    Computing initial clusters ...
    merging clusters ...
    assigning cluster num to each point ...
    Computing cluster centers ...
    time for k=3 clusters: 19.26
    


    
![png](readme_notebook_files/readme_notebook_49_3.png)
    


    Processing k=4:
    Computing initial clusters ...
    merging clusters ...
    assigning cluster num to each point ...
    Computing cluster centers ...
    time for k=4 clusters: 18.78
    


    
![png](readme_notebook_files/readme_notebook_49_5.png)
    

#### Files Structure
|Task4_P1_thresholding(Global-Local).ipynb| problem 1  global and local thresholding techniques [optimal - otsu - spectral(multilevel)] |
|-------------------|----------------|
|Task4_P2(Segmentation).ipynb| problem 2  some segmentation techniques [k_mean - mean_shift  - region_growing - agglomerative] |
|task4_util.py| problem 1 helper functions|
|task4_util_segmentation.py| problem 2 helper functions|