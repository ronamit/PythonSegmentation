# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 08:14:09 2017

@author: Ron
"""
from __future__ import division
import cv2
import numpy as np
import scipy.cluster as spc
import scipy.ndimage.morphology as spm
import matplotlib.pyplot as plt


def calcKMeans(points, K):
    """Calculate KMeans."""

    if len(points) < K:
        return False, None, None

  # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    term_crit = (cv2.TERM_CRITERIA_EPS , 30, 0.1)
#    term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Apply KMeans
    ret,labels,centers = cv2.kmeans(points, K, None, term_crit, 10, cv2.KMEANS_RANDOM_CENTERS)

    #
    # Evaluate success.
    # Note:
    # If one of the segments is less than 5% we assume
    # that the K was too big and therefore failed.
    #
    
    #@ Debug: labels
#    print  'Labels: ', np.unique(labels)
    plt.hist(labels, 'auto') 
    plt.title("Histogram of labels")
    plt.show()
    
#    for i in range(K):
#        print 'Number of labels with label ', i, ' is ', (labels==i).sum()
    
    #@
    
    minRatio = 0.01
    for i in range(K):
        pixRatioInCluster =  (labels==i).sum()/labels.size
#        print pixRatioInCluster
        if pixRatioInCluster < minRatio:
            return False, None, None

    return True, labels, centers


#################### Script start
#@ Load image
# imagePath = 'Images/manual_006_aa77fc74890d_crop.jpg'
# imagePath = 'Images/Selection_002.png'
#imagePath = 'Images/Crop2.jpg'
imagePath = 'Images/manual_004_106160955a15_crop.jpg'
crop = cv2.imread(imagePath)
#@

#@ Display image
plt.imshow(crop)
plt.show()
#@


# Calculate K kmeans: bg color, fg color, letter color
# I use the LAB color space.

lab = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
lab = cv2.medianBlur(lab, 7)

#@ Display image
cv2.namedWindow("Image in LAB after blur", cv2.WINDOW_NORMAL)
cv2.imshow("Image in LAB after blur", lab)
#@

lab_points = lab.reshape(-1, 3).astype(np.float32)

K = 4
while K > 1:
    print 'Try K = ', K
    success, labels, centers = calcKMeans(lab_points, K)
    if success:
        print 'success with K = ', K
        break
    K -= 1

if not success:
   print 'Output: None'
else:
    #
    # Use the normalised moments to identify the fg (shape) mask.
    # The target is in the center and has the minimum normalized
    # moments of order nu20, nu02.
    #
    kernel = np.ones((3, 3), np.uint8)
    mom = []
    bin_imgs = []
    for i in range(K):
        bin_img = np.zeros(shape=crop.shape[:2], dtype=np.uint8)
        bin_img.flat = labels == i
        
        #@ Debug
#        print 'Label ', i, ' mask'
#        plt.matshow(bin_img)
#        plt.title('Label {0} mask'.format(i))
#        plt.show()
        #@
        
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
        
        #@ Debug
#        plt.matshow(bin_img)
#        plt.title('Label {0} mask after morpholical open'.format(i))
#        plt.show()
        #@
        
        bin_imgs.append(bin_img)
        moments = cv2.moments(bin_img, True)
        momentsValue  = (moments['nu20'] + moments['nu02'])*moments['m00']
        mom.append(momentsValue)
        
        #@ Debug 
        print 'Moments value of mask label ', i, ' is ', momentsValue
        #@
    # End for
    order = np.argsort(mom)
    # The shape is usually the second in the order
    # but just in case, we verify that its area is
    # bigger as sometimes the letter color matches
    # a very noisy surrounding.
    #

    im0 =  bin_imgs[order[0]].copy()
    im1 =  bin_imgs[order[1]].copy()
    
    # @ Debug
    plt.matshow(im0)
    plt.title('Mask with smallest  moment ')
    plt.show()
    # @
       
    # @ Debug
    plt.matshow( bin_imgs[order[1]])
    plt.title('Mask with second smallest  moment ')
    plt.show()
    # @
    
    
    imTemp, contours, hierarchy  = cv2.findContours( im0, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#    print 'Test ', cv2.contourArea(contours[0])
    
    if len(contours) == 0:
        area0 = 0
    else:
        contours0 = max(contours, key=cv2.contourArea)
        area0 = cv2.contourArea(contours0)
    print 'Area of Mask with smallest  moment: ', area0

    imTemp, contours, hierarchy  = cv2.findContours(im1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        area1 = 0
    else:
        contours1 = max(contours, key=cv2.contourArea)
    area1 = cv2.contourArea(contours1)
    
    print 'Area of Mask with second smallest  moment: ', area1

    if area0 > area1:
        shape_index = 0
    else:
        shape_index = 1

    shape = bin_imgs[order[shape_index]]
    
       #
    # Fill the letter hole in the shape.
    # Note:
    # I first try to connect broken
    # lines
    #
    kernel = np.ones((3, 3), np.uint8)
    shape = cv2.filter2D(shape, -1, kernel)
    shape[shape>0] = 1
    imTemp, contours, hierarchy = cv2.findContours(shape.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:
        max_contour = max(contours, key=cv2.contourArea)
    except:
       print ' return (None, None)'

    filled = np.zeros_like(shape)
    cv2.fillPoly(filled, [np.squeeze(max_contour)], (1,))
    filled = cv2.erode(filled, kernel, iterations=3)
    
   # @ Debug
    plt.matshow(filled)
    plt.title('Filled shape mask ')
    plt.show()
    # @
     
     #
    # Segment the shape to 2 colors: fg and letter
    #
    fg_indices = filled.flat == 1
    success, labels, centers = calcKMeans(lab_points[fg_indices], K=2)
    if not success:
        print ' return (None, None)'

    #
    # Use the number of pixels to identify the fg mask.
    # Usually the letter is smaller than the shape.
    #
    bin_imgs = []
    moments = []
    for i in range(2):
        bin_img = np.zeros(shape=crop.shape[:2], dtype=np.uint8)
        bin_img.flat[fg_indices] = labels == i
        bin_imgs.append(bin_img)

        m = cv2.moments(bin_img, True)
        moments.append(m['m00'])

    #
    # Identify the letter, fg masks
    #

    letter_index = np.argmin(moments)
    letter_mask = bin_imgs[letter_index]
    letter_mask = letter_mask*255
        

   # @ Debug
    plt.matshow(letter_mask)
    plt.title('Final letter mask ')
    plt.show()
    # @

   

print 'End'
#@