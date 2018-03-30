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
input_image_full = lab.imread("Shapes.png")

# Resize it to 10% of the original size to speed up the processing
input_image = sp.misc.imresize(input_image_full, 0.2) / 255.

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
                if dist >= 1:
                    print(dist)
                graph[first_idx, second_idx] = dist
                graph[second_idx, first_idx] = dist

# Apply spectral clustering (this step goes much faster if you have pyamg
# installed)
N_REGIONS = 7 

#############################################################################
# Visualize the resulting regions

for assign_labels in ('kmeans', 'discretize'):
    t0 = time.time()
    labels = spectral_clustering(graph, n_clusters=N_REGIONS,
                                 assign_labels=assign_labels, random_state=1)

    t1 = time.time()
    labels = labels.reshape((input_image.shape[0], input_image.shape[1]))

    plt.figure(figsize=(5, 5))
    plt.imshow(input_image)
    for l in range(N_REGIONS):
        plt.contour(labels == l, contours=1,
                    colors=[plt.cm.spectral(l / float(N_REGIONS))])
    plt.xticks(())
    plt.yticks(())
    title = 'Spectral clustering: %s, %.2fs' % (assign_labels, (t1 - t0))
    print(title)
    plt.title(title)
plt.show()
