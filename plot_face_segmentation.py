"""
===================================================
Segmenting the picture of a raccoon face in regions
===================================================

This example uses :ref:`spectral_clustering` on a graph created from
voxel-to-voxel difference on an image to break this image into multiple
partly-homogeneous regions.

This procedure (spectral clustering on an image) is an efficient
approximate solution for finding normalized graph cuts.

There are two options to assign labels:

* with 'kmeans' spectral clustering will cluster samples in the embedding space
  using a kmeans algorithm
* whereas 'discrete' will iteratively search for the closest partition
  space to the embedding space.
"""

# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>, Brian Cheung
# License: BSD 3 clause

import time
import math

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.pylab as lab

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from sklearn.cluster import AgglomerativeClustering 

# Read in the image
input_image_full = lab.imread("Stop.jpeg")

# Resize it to 10% of the original size to speed up the processing
input_image = sp.misc.imresize(input_image_full, 0.9) / 255.

'''
graph = np.zeros((input_image.shape[0] * input_image.shape[1], input_image.shape[0] * input_image.shape[1] ))
for i in range (0, input_image.shape[0]):
    for j in range (0, input_image.shape[1]):
        for i2 in range(i, input_image.shape[0]):
            for j2 in range(i, input_image.shape[1]):
                dist_sum = 0
                for c in range (0, input_image.shape[2]):
                    dist_sum += (input_image[i2, j2, c] - input_image[i, j, c])**2 
                first_idx = i * input_image.shape[1] + j
                second_idx = i2 * input_image.shape[1] + j2
                
                dist = math.sqrt(dist_sum) / math.sqrt(3)
                graph[first_idx, second_idx] = dist
                graph[second_idx, first_idx] = dist
'''

input_image_gray = np.zeros((input_image.shape[0], input_image.shape[1]))

for i in range (0, input_image.shape[0]):
    for j in range (0, input_image.shape[1]):
        input_image_gray[i, j] = 0
        '''
        input_image_gray[i, j] += input_image[i, j, 2] * 0.299
        input_image_gray[i, j] += input_image[i, j, 1] * 0.587
        input_image_gray[i, j] += input_image[i, j, 0] * 0.114
        '''
        
        input_image_gray[i, j] += input_image[i, j, 0] * 0.2125
        input_image_gray[i, j] += input_image[i, j, 1] * 0.7154
        input_image_gray[i, j] += input_image[i, j, 2] * 0.0721

print("GRAYSCALED")

graph = image.img_to_graph(input_image_gray)

print("GRAPHED")

# Take a decreasing function of the gradient: an exponential
# The smaller beta is, the more independent the segmentation is of the
# actual image. For beta=1, the segmentation is close to a voronoi
beta = 5 
eps = 1e-6
graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps

print("EXPONENTIALED")

# Apply spectral clustering (this step goes much faster if you have pyamg
# installed)
N_REGIONS = 25 

#############################################################################
# Visualize the resulting regions

t0 = time.time()
labels = spectral_clustering(graph, n_clusters=N_REGIONS, random_state=1)

t1 = time.time()
labels = labels.reshape((input_image.shape[0], input_image.shape[1]))

plt.figure(figsize=(5, 5))
plt.imshow(input_image, cmap=plt.cm.gray)
for l in range(N_REGIONS):
    plt.contour(labels == l, contours=1,
                colors=[plt.cm.spectral(l / float(N_REGIONS))])
plt.xticks(())
plt.yticks(())
title = 'Spectral clustering: %s, %.2fs' % ("kmeans", (t1 - t0))
print(title)
plt.title(title)
plt.show()


'''
# #############################################################################
# Compute clustering
print("Compute structured hierarchical clustering...")
st = time.time()
ward = AgglomerativeClustering(n_clusters=N_REGIONS, linkage='ward')
                               
ward.fit(graph)
label = np.reshape(ward.labels_, (input_image.shape[0], input_image.shape[1]))
print("Elapsed time: ", time.time() - st)
print("Number of pixels: ", label.size)
print("Number of clusters: ", np.unique(label).size)

# #############################################################################
# Plot the results on an image
plt.figure(figsize=(5, 5))
plt.imshow(input_image)
for l in range(N_REGIONS):
    plt.contour(label == l, contours=1,
                colors=[plt.cm.spectral(l / float(N_REGIONS)), ])
plt.xticks(())
plt.yticks(())
plt.show()
'''
