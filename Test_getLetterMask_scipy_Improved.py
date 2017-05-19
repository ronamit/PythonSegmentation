
from __future__ import division
import cv2
import numpy as np
import scipy.cluster as spc
import scipy.ndimage.morphology as spm
import matplotlib.pyplot as plt

def innerMost(R, shape_colors_map, shape_indices):
    # Check for which label (0 or 1) the mean distance from center is smaller
    meanDist0 = R.ravel()[shape_indices[shape_colors_map == 0]].mean() 
    meanDist1 = R.ravel()[shape_indices[shape_colors_map == 1]].mean() 
    print meanDist0
    print meanDist1
    
    if meanDist0 < meanDist1:
        inner_index = 0
    else:
        inner_index = 1

    return inner_index


#################### Script start

#@ Load image
#imagePath = 'Images/manual_007_40d11814c73d.jpg'
#imagePath = 'Images/Selection_002.png'
imagePath = 'Images/Crop2.jpg'
#imagePath = 'Images/manual_004_106160955a15_crop.jpg'
img = cv2.imread(imagePath)
#@


width, height = img.shape[:2]

##@ Display image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # flip RGB channels
#@


# Divide the crop to two colors. (expected to separate: shape and background)
#
img_vectors = img.reshape((-1, 3)).astype(np.float)
colors, dist = spc.vq.kmeans(img_vectors, 2)
color_map, _ = spc.vq.vq(img_vectors, colors)


# Calculate distance map.
#
X, Y = np.mgrid[:width, :height] 
R = np.sqrt((X - width / 2) ** 2 + (Y - height / 2) ** 2)

#@ Display shape mask
print 'Distance matrix'
plt.matshow(R)
plt.show()
#@


#
# Assuming a tight crop the target should occupy most of the crop.
#
img_indices = np.arange(width * height)
shape_index = innerMost(R, color_map, img_indices)
print shape_index

#@ Display separation
print 'k-means with k=2 by colors:'
plt.matshow(color_map.reshape(width, height) == shape_index)
plt.show()
#@


#
# Smooth the map to remove noise.
# Erode it to remove the border of the shape the might damage the segmentation to shape and letter.
#

shape_mask = np.zeros(img.shape[:2], dtype=np.uint8)
shape_mask.ravel()[color_map == shape_index] = 1
shape_mask = cv2.medianBlur(shape_mask, ksize=15)
kernel = np.ones((5, 5), np.uint8)
shape_mask = cv2.erode(shape_mask, kernel, iterations=1)

#@ Display shape mask
print 'shape mask - after smooth and erode  (remove border):'
plt.matshow(shape_mask)
plt.show()
#@

#
# Fill holes (incase the color of the letter is similar to the background.)

shape_mask = spm.binary_fill_holes(shape_mask)

#@ Display shape mask
print 'shape mask - after filling holes:'
plt.matshow(shape_mask)
plt.show()
#@

#
#
#
shape_indices = np.arange(shape_mask.size)[shape_mask.ravel() == 1]

#
# Split the shape in two colors.
#
shape_vectors = img_vectors[shape_mask.ravel() == 1]
shape_colors, dist = spc.vq.kmeans(shape_vectors, 2)
shape_colors_map, _ = spc.vq.vq(shape_vectors, shape_colors)
#logging.info('Classified shape colors: {}'.format(shape_colors))


#
# The letter should be the innermost
#
letter_index = innerMost(R, shape_colors_map, shape_indices)
#logging.info('Classified letter color index: {}'.format(letter_index))

 #
# Letter indices
#
letter_indices = shape_indices[shape_colors_map == letter_index]
neto_shape_indices = shape_indices[shape_colors_map == (1-letter_index)]
rawLetterMask = np.zeros(img.shape[:2], dtype=np.uint8)
rawLetterMask.ravel()[letter_indices] = 255
neto_shape_mask = np.zeros(img.shape[:2], dtype=np.uint8) # Shpae minus letter 
neto_shape_mask.ravel()[neto_shape_indices] = 0  # ATODO


#@ Display letter mask
print 'Raw Letter mask (before connected componrnnets separate):'
plt.matshow(rawLetterMask)
plt.show()
#@




# You need to choose 4 or 8 for connectivity type
connectivity = 4  
# Perform the operation
output = cv2.connectedComponentsWithStats(rawLetterMask, connectivity, cv2.CV_32S)
# Get the results
# The first cell is the number of labels
num_labels = output[0]
# The second cell is the label matrix
labels = output[1]
# The third cell is the stat matrix
stats = output[2]
# The fourth cell is the centroid matrix
centroids = output[3]


areasList  = stats[:, cv2.CC_STAT_AREA]
orederByArea = np.argsort(areasList)
labelOfMax = orederByArea[-2] # Since the largest componenet is the BG, we take the second largest

letter_mask = 255*(labels == labelOfMax)


#@ Display letter mask
print 'Final letter mask'
plt.matshow(letter_mask)
plt.show()
#@



print 'End'
#@

