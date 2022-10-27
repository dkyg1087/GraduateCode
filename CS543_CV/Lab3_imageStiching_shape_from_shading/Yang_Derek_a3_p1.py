# imports
import numpy as np
import skimage
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance
import scipy 
import random as rd

##############################################
### Provided code - nothing to change here ###
##############################################

"""
Harris Corner Detector
Usage: Call the function harris(filename) for corner detection
Reference   (Code adapted from):
             http://www.kaij.org/blog/?p=89
             Kai Jiang - Harris Corner Detector in Python
             
"""
from pylab import *
from scipy import signal
from scipy import *
import numpy as np
from PIL import Image

def harris(filename, min_distance = 10, threshold = 0.1):
    """
    filename: Path of image file
    threshold: (optional)Threshold for corner detection
    min_distance : (optional)Minimum number of pixels separating 
     corners and image boundary
    """
    im = np.array(Image.open(filename).convert("L"))
    harrisim = compute_harris_response(im)
    filtered_coords = get_harris_points(harrisim,min_distance, threshold)
    plot_harris_points(im, filtered_coords)

def gauss_derivative_kernels(size, sizey=None):
    """ returns x and y derivatives of a 2D 
        gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    y, x = mgrid[-size:size+1, -sizey:sizey+1]
    #x and y derivatives of a 2D gaussian with standard dev half of size
    # (ignore scale factor)
    gx = - x * exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2))) 
    gy = - y * exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2))) 
    return gx,gy

def gauss_kernel(size, sizey = None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = mgrid[-size:size+1, -sizey:sizey+1]
    g = exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()

def compute_harris_response(im):
    """ compute the Harris corner detector response function 
        for each pixel in the image"""
    #derivatives
    gx,gy = gauss_derivative_kernels(3)
    imx = signal.convolve(im,gx, mode='same')
    imy = signal.convolve(im,gy, mode='same')
    #kernel for blurring
    gauss = gauss_kernel(3)
    #compute components of the structure tensor
    Wxx = signal.convolve(imx*imx,gauss, mode='same')
    Wxy = signal.convolve(imx*imy,gauss, mode='same')
    Wyy = signal.convolve(imy*imy,gauss, mode='same')   
    #determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy   
    return Wdet / Wtr

def get_harris_points(harrisim, min_distance=10, threshold=0.1):
    """ return corners from a Harris response image
        min_distance is the minimum nbr of pixels separating 
        corners and image boundary"""
    #find top corner candidates above a threshold
    corner_threshold = max(harrisim.ravel()) * threshold
    harrisim_t = (harrisim > corner_threshold) * 1    
    #get coordinates of candidates
    candidates = harrisim_t.nonzero()
    coords = [ (candidates[0][c],candidates[1][c]) for c in range(len(candidates[0]))]
    #...and their values
    candidate_values = [harrisim[c[0]][c[1]] for c in coords]    
    #sort candidates
    index = argsort(candidate_values)   
    #store allowed point locations in array
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_distance:-min_distance,min_distance:-min_distance] = 1   
    #select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i][0]][coords[i][1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i][0]-min_distance):(coords[i][0]+min_distance),
                (coords[i][1]-min_distance):(coords[i][1]+min_distance)] = 0               
    return filtered_coords

def plot_harris_points(image, filtered_coords):
    """ plots corners found in image"""
    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'r*')
    axis('off')
    show()

# Usage: 
#harris('./path/to/image.jpg')


# Provided code for plotting inlier matches between two images

def plot_inlier_matches(ax, img1, img2, inliers):
    """
    Plot the matches between two images according to the matched keypoints
    :param ax: plot handle
    :param img1: left image
    :param img2: right image
    :inliers: x,y in the first image and x,y in the second image (Nx4)
    """
    res = np.hstack([img1, img2])
    ax.set_aspect('equal')
    ax.imshow(res, cmap='gray')
    
    ax.plot(inliers[:,0], inliers[:,1], '+r')
    ax.plot(inliers[:,2] + img1.shape[1], inliers[:,3], '+r')
    ax.plot([inliers[:,0], inliers[:,2] + img1.shape[1]],
            [inliers[:,1], inliers[:,3]], 'r', linewidth=0.4)
    ax.axis('off')
    
# Usage:
# fig, ax = plt.subplots(figsize=(20,10))
# plot_inlier_matches(ax, img1, img2, computed_inliers)


#######################################
### Your implementation starts here ###
#######################################

# See assignment page for the instructions!
def compute_error(H,matches):
    num_pairs = len(matches)
    
    p1 = np.concatenate((matches[:,0:2],np.ones((1,num_pairs)).T),axis = 1)
    p2 = matches[:,2:4]
    
    t_p1 = np.zeros((num_pairs,2))
    
    for i in range(num_pairs):
        #print(H.shape,len(p1[i]))
        t_p1[i] = (np.matmul(H, p1[i]) / np.matmul(H, p1[i])[-1])[0:2]
    
    return np.linalg.norm(p2 - t_p1, axis=1) ** 2

def compute_homo(subset):
    A = []

    for i in range(subset.shape[0]):
        
        p1 = np.append(subset[i][0:2], 1)
        p2 = np.append(subset[i][2:4], 1)
        
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
        
        A.append(row1)
        A.append(row2)

    U, s, V = np.linalg.svd(np.array(A))
    
    H = V[len(V)-1].reshape(3, 3)

    H = H / H[2, 2]
    
    return H
        

def ransac(img1,img2,matches,threshold):
    inliers = 0
    best_inliers = 0
    best_H = []
    best_inliers_pt = []
    
    for i in range(1000):
        idx = rd.sample(range(matches.shape[0]),4)
        subset = matches[idx]
        
        H = compute_homo(subset)
        
        if np.linalg.matrix_rank(H) < 3:
            continue
        
        error = compute_error(H,matches)
        mask = np.where(error < threshold)[0]
        inliers_pt  = matches[mask]
        
        inliers = len(inliers_pt)
        if inliers > best_inliers:
            best_inliers = inliers
            best_H = H.copy()
            best_inliers_pt = inliers_pt.copy()

            avg_residual = sum(compute_error(best_H,best_inliers_pt)) / best_inliers

    print("Inliers:",best_inliers,"Average residuals:",avg_residual)
    
    fig,ax = plt.subplots(figsize=(20,10))
    #plot_inlier_matches(ax, img1, img2, best_inliers_pt)
    #plt.savefig("plot_basic.jpg")
    
    return best_H

def distance_calculation(kp1,dsp1,kp2,dsp2,threshold):
    dist = scipy.spatial.distance.cdist(dsp1, dsp2, 'sqeuclidean')

    idx1 = np.where(dist < threshold)[0]
    idx2 = np.where(dist < threshold)[1]
    
    cd1 = np.array([kp1[idx].pt for idx in idx1])
    cd2 = np.array([kp2[idx].pt for idx in idx2])
    
    matches = np.concatenate((cd1, cd2), axis=1)

    return matches

def warp_and_stitch(img1,img2,H):
    transform = skimage.transform.ProjectiveTransform(H)
    warp = skimage.transform.warp

    h,w,_ = img1.shape

    c_temp = np.array([[0, 0],
                        [0, h],
                        [w, 0],
                        [w, h]])

    c_w = transform(c_temp)
    corner = np.vstack((c_w, c_temp))

    c_min = np.min(corner, axis=0)
    c_max = np.max(corner, axis=0)

    output_shape = (c_max - c_min)
    output_shape = np.ceil(output_shape[::-1])

    offset = skimage.transform.SimilarityTransform(translation=-c_min)

    img1_w = warp(img1, (transform + offset).inverse, output_shape=output_shape, cval=0)
    img2_w = warp(img2, offset.inverse, output_shape=output_shape, cval=0)

    img2_w[img1_w > 0] = 0
    result = img1_w + img2_w 
    result = cv2.convertScaleAbs(result, alpha=(255.0))
    return result


def stitch_image(img1,img2):
    img1_g = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2_g = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    
    kp1,dsp1 = cv2.SIFT_create().detectAndCompute(img1_g,None)
    kp2,dsp2 = cv2.SIFT_create().detectAndCompute(img2_g,None)
    
    matches = distance_calculation(kp1,dsp1,kp2,dsp2,9000)
    
    H = ransac(img1_g,img2_g,matches,1.5)
    
    stitched = warp_and_stitch(img1,img2,H)
    
    return stitched

def main():
    #img_dir = 'data/'
    img_dir = 'data/ledge/'
    
    #img_name = ['right.jpg','left.jpg']
    img_name = ['3.jpg','2.jpg','1.jpg']
    img_list = []
    
    for name in img_name:
        img_list.append(cv2.imread(img_dir + name))
    
    while len(img_list) > 1 :
        img1 = img_list.pop()
        img2 = img_list.pop()
        stitched_img = stitch_image(img1,img2)
        img_list.insert(0,stitched_img)


    cv2.imshow("result",img_list[0])
    cv2.imwrite("result_ledge.jpg",img_list[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()