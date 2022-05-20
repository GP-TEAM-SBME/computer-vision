[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=7209332&assignment_repo_type=AssignmentRepo)


# TASK 1 [Problem `1` to Problem `6` ]
>Note:  In this submission we implemented our utils functions from scratch in python, like what was asked in cpp, but according to the new guidlines we will vectorize our functions utilizing numpy in the final submission ISA.

> the histogram was implemented from scratch 
> - `draw_histogram_scratch` in problem 4 and all visualizations of histograms 
> - `get_histogram` in problem 5 & 6 which is used in all the calculations 
>
> we will remove this dublication ISA


# Problem 1 [Adding Additive Noise]

## Uniform Noise
<p align="center">
  <img src="output/uniform_noise.png" width="500", height="200" />   
</p>
 
 ## Salt and Pepper Noise
<p align="center">
  <img src="output/salt_pepper_noise.png" width="500", height="200" />   
</p>
 
 ## Gaussian Noise
<p align="center">
  <img src="output/gaussian_noise.png" width="500", height="200" />   
</p>
 


 # Problem 2 [Filtering Noise]

## Average Filter
<p align="center">
  <img src="output/average_filter.png" width="500", height="200" />   
</p>
 
 ## Median
<p align="center">
  <img src="output/salt_pepper_filtered.png" width="500", height="200" />   
</p>
 
 ## Gaussian Noise
<p align="center">
  <img src="output/gaussian_filter.png" width="500", height="200" />   
</p>


# Problem 3 [Edge Detection (Sobel , Roberts , Prewitt and Canny) ]

## Sobel , Roberts , Prewitt
<p align="center">
  <img src="output/edges.png" width="500", height="200" />   
</p>
 
## Canny Edges
<p align="center">
  <img src="output/canny.png" width="300", height="200" />   
</p>
 
# Problem 4 [Histogram and Distribution Curves]

<p align="center">
  <img src="output/hist.png" width="500", height="200" />   
</p>
 
<p align="center">
  <img src="output/hist_kde.png" width="500", height="200" />   
</p>
 
 <p align="center">
  <img src="output/hist_g.png" width="500", height="200" />   
</p>

# Problem 5 [Equalizing Images]

## Cummulative sum for generating image CDF used in equalization
<p align="center">
  <img src="output/cdf.png" width="500", height="200" />   
</p>

## Image histograms before and after equalization
<p align="center">
  <img src="output/equalization_hist.png" width="500", height="200" />   
</p>
 
 <p align="center">
  <img src="output/equalized_lung_hist.png" width="500", height="200" />   
</p>


## Images before and after equalization (contrast enhancement)
<p align="center">
  <img src="output/equalized_lena.png" width="500", height="200" />   
</p>
 
 <p align="center">
  <img src="output/equalized_lung.png" width="500", height="200" />   
</p>


## Problem 6 [Normalizing Images]
<p align="center">
  <img src="output/norm_hist.png" width="500", height="200" />   
</p>
 
 <p align="center">
  <img src="output/norm_img.png" width="500", height="200" />   
</p>