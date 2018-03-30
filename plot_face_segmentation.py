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

# Read in the image
input_image = lab.imread("Shapes.png")

# Resize it to 10% of the original size to speed up the processing
input_image = sp.misc.imresize(input_image, 0.25) / 255.

print(input_image.shape)
print(input_image[0, 0])
print(input_image[1, 0])

dist_matrix = np.zeros((input_image.shape[0] * input_image.shape[1], input_image.shape[0] * input_image.shape[1] ))
for i in range (0, input_image.shape[0]):
    for j in range (0, input_image.shape[1]):
        for i2 in range(i, input_image.shape[0]):
            for j2 in range(i, input_image.shape[1]):
                dist_sum = 0
                for c in range (0, input_image.shape[2]):
                    dist_sum += (input_image[i2, j2, c] - input_image[i, j, c])**2 
                first_idx = i * input_image.shape[0] + j
                second_idx = i2 * input_image.shape[0] + j2
                
                dist_matrix[first_idx, second_idx] = math.sqrt(dist_sum)

print("DISTANCE")
print(dist_matrix[0, 2])

        
plt.imshow(input_image)
plt.show()

'''

# Convert the image into a graph with the value of the gradient on the
# edges.
graph = image.img_to_graph(input_image)

# Take a decreasing function of the gradient: an exponential
# The smaller beta is, the more independent the segmentation is of the
# actual image. For beta=1, the segmentation is close to a voronoi
beta = 5
eps = 1e-6
graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps

# Apply spectral clustering (this step goes much faster if you have pyamg
# installed)
N_REGIONS = 5

#############################################################################
# Visualize the resulting regions

for assign_labels in ('kmeans', 'discretize'):
    t0 = time.time()
    labels = spectral_clustering(graph, n_clusters=N_REGIONS,
                                 assign_labels=assign_labels, random_state=1)
    t1 = time.time()
    labels = labels.reshape(input_image.shape)

    plt.figure(figsize=(5, 5))
    plt.imshow(input_image, cmap=plt.cm.gray)
    for l in range(N_REGIONS):
        plt.contour(labels == l, contours=1,
                    colors=[plt.cm.spectral(l / float(N_REGIONS))])
    plt.xticks(())
    plt.yticks(())
    title = 'Spectral clustering: %s, %.2fs' % (assign_labels, (t1 - t0))
    print(title)
    plt.title(title)
plt.show()
'''
